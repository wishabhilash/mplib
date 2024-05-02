import datetime as dt

try:
    import pandas as pd
except ImportError:
    pass

import os
from .. import utils
from .session import FyersSession, ShoonyaSession
from ..shoonya import *
from pydantic import BaseModel, Field


class Historical:
    def __init__(self, session: FyersSession) -> None:
        self._session = session
        self._client = None

    def historical(self, symbol, resolution, start, end):
        curr = start
        delta = dt.timedelta(days=100)
        final_data = []
        if self._client is None:
            self._client = self._session.init_client()
        
        while curr < end:
            payload = {
                "symbol": symbol,
                "resolution": f"{resolution}",
                "date_format": "1",
                "range_from": f'{curr:%Y-%m-%d}',
                "range_to": f'{(curr + delta):%Y-%m-%d}',
                "cont_flag": "1",
            }
            
            data = self._client.history(data=payload)
            final_data += data["candles"]
            curr += delta + dt.timedelta(days=1)
        df = pd.DataFrame(final_data, columns=["datetime", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df.index = df.index.tz_convert("Asia/Kolkata")
        df.datetime = df.index
        df = df.sort_index()
        return df

class Position(BaseModel):
    symbol: str = Field(..., alias='tysm')
    quantity: str = Field(..., alias='netqty')
    product_type: str = Field(..., alias='prd')


class BaseBroker:
    def buy(self, symbol, qty=1, price=None):
        raise NotImplementedError

    def sell(self, symbol, qty=1, price=None):
        raise NotImplementedError
    
    def positions(self, symbol, product_type):
        raise NotImplementedError

    def orders(self):
        raise NotImplementedError


class ShoonyaBroker(BaseBroker):
    _positions = None
    _limits = None

    def __init__(self, session: ShoonyaSession, polling_interval=1) -> None:
        self._session = session
        self._client = self._session.init_client()
        self._polling_interval = polling_interval

        self._get_positions()

    def _get_symbol(self, term=None, exchange='NSE', instrument_type='EQ'):
        d = self._client.searchscrip(exchange, term)
        for i in d['values']:
            if i['instname'] == instrument_type:
                return i['tsym']
        return None

    def _order(self, symbol, qty=1, price=0.0, product_type=None, direction=None):
        exchange, term = symbol.split(':')
        tradingsymbol = self._get_symbol(term, exchange)
        
        res = self._client.place_order(
            buy_or_sell=direction,
            product_type=ProductType.Intraday if product_type is None else product_type,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=qty,
            discloseqty=0,
            price_type=PriceType.Limit if price != 0.0 else PriceType.Market,
            price=price,
        )
        if res['stat'] == 'Ok':
            return res['norenordno']
        return None

    def buy(self, symbol, qty=1, price=None):
        return self._order(symbol, qty, price, direction=BuyorSell.Buy)
    
    def sell(self, symbol, qty=1, price=0.0, product_type=None):
        return self._order(symbol, qty, price, direction=BuyorSell.Sell)
    
    def coverorder(self, symbol, price, sl, qty=1):
        exchange, term = symbol.split(':')
        tradingsymbol = self._get_symbol(term, exchange)
        
        res = self._client.place_order(
            buy_or_sell=BuyorSell.Sell,
            product_type=ProductType.CO,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=qty,
            discloseqty=0,
            price_type=PriceType.Limit,
            price=price,
            bookloss_price=sl
        )
        if res['stat'] == 'Ok':
            return res['norenordno']
        return None

    def bracketorder(self, symbol, price, sl, tp, direction, qty=1):
        exchange, term = symbol.split(':')
        tradingsymbol = self._get_symbol(term, exchange)
        
        res = self._client.place_order(
            buy_or_sell=direction,
            product_type=ProductType.BO,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=qty,
            discloseqty=0,
            price_type=PriceType.Limit if price != 0 else PriceType.Market,
            price=price,
            bookloss_price=sl,
            bookprofit_price=tp
        )
        if res['stat'] == 'Ok':
            return res['norenordno']
        return None

    def bobuy(self, symbol, price, sl, tp, qty=1):
        return self.bracketorder(symbol, price, sl, tp, BuyorSell.Buy, qty)

    def bosell(self, symbol, price, sl, tp, qty=1):
        return self.bracketorder(symbol, price, sl, tp, BuyorSell.Sell, qty)

    def exitorder(self, order_number):
        return self._client.exit_order(orderno=order_number)
    
    @utils.threaded
    def _get_positions(self, poll_interval=5):
        while True:
            p = self._client.get_positions()
            self._positions = pd.DataFrame(p)
            time.sleep(self._polling_interval)

    def positions(self, symbol, product_type=ProductType.BO):
        result = self._positions[(self._positions.tsym == symbol) & (self._positions.prd == product_type) & (self._positions.netqty > 0)]
        if not len(result.list()):
            return None
        return [ Position.model_validate(i) for i in result.to_dict('records')]

    def orders(self):
        raise NotImplementedError

    def limits(self):
        return self._limits