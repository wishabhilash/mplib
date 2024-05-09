import datetime as dt

try:
    import pandas as pd
except ImportError:
    pass

from retry import retry
from .. import utils
from .session import FyersSession, ShoonyaSession
from .shoonya import *
from pydantic import BaseModel, Field


class Historical:
    def __init__(self, session: FyersSession) -> None:
        self._session = session
        self._client = None

    @retry(tries=5, delay=2)
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
            try:
                final_data += data["candles"]
            except IndexError as e:
                continue
            curr += delta + dt.timedelta(days=1)
        df = pd.DataFrame(final_data, columns=["datetime", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df.index = df.index.tz_convert("Asia/Kolkata")
        df.datetime = df.index
        df = df.sort_index()
        return df

class Position(BaseModel):
    symbol: str = Field(..., alias='tsym')
    quantity: int = Field(..., alias='netqty')
    product_type: str = Field(..., alias='prd')

class PositionExt(Position):
    price: float = None
    tp_target: float = None
    sl_target: float = None
    direction: str = None
    datetime: dt.datetime = None

class Tick(BaseModel):
    datetime: dt.datetime
    ltp: float
    symbol: str

class Order(BaseModel):
    entry_time: dt.datetime
    entry_price: float
    sl: float
    tp: float
    direction: int
    exit_time: dt.datetime = None
    exit_price: float = None
    profit: float = None

    def set_profit(self):
        if self.direction == 1:
            self.profit = self.exit_price - self.entry_price
        else:
            self.profit = self.entry_price - self.exit_price

class BaseBroker:
    def buy(self, symbol, qty=1, price=None):
        raise NotImplementedError

    def sell(self, symbol, qty=1, price=None):
        raise NotImplementedError
    
    def positions(self, symbol, product_type):
        raise NotImplementedError
    
    def coverorder(self, symbol, price, sl, qty=1):
        raise NotImplementedError

    def bracketorder(self, symbol, price, sl, tp, direction, qty=1):
        raise NotImplementedError
    
    def bobuy(self, symbol, price, sl, tp, qty=1):
        raise NotImplementedError

    def bosell(self, symbol, price, sl, tp, qty=1):
        raise NotImplementedError
    
    def notify_tick(self, tick: Tick):
        raise NotImplementedError


class MockBroker(BaseBroker):
    __positions: dict = {}
    __ticks: dict = {}
    __orders: list = []

    def _get_position_key(self, pos: Position):
        return f'{pos.symbol}_{pos.product_type}'

    def _add_position(self, pos: Position):
        self.__positions[self._get_position_key(pos)] = pos

    def buy(self, symbol, qty=1, price=None):
        curr_tick = self.__ticks[symbol]
        p = PositionExt(
            tsym=symbol,
            netqty=qty,
            prd=ProductType.Intraday,
            price=curr_tick.ltp,
            direction=BuyorSell.Buy
        )
        self._add_position(p)

    def sell(self, symbol, qty=1, price=None):
        curr_tick = self.__ticks[symbol]
        p = PositionExt(
            tsym=symbol,
            netqty=qty,
            prd=ProductType.Intraday,
            price=curr_tick.ltp,
            direction=BuyorSell.Buy
        )
        self._add_position(p)
    
    def positions(self, symbol, product_type):
        pos = Position(tsym=symbol, netqty=0, prd=product_type)
        k = self._get_position_key(pos)
        return self.__positions[k] if k in self.__positions else None
    
    def bobuy(self, symbol, price, sl, tp, qty=1):
        self.bracketorder(symbol, price, sl, tp, BuyorSell.Buy, qty)

    def bosell(self, symbol, price, sl, tp, qty=1):
        self.bracketorder(symbol, price, sl, tp, BuyorSell.Sell, qty)
    
    def bracketorder(self, symbol, price, sl, tp, direction, qty=1):
        curr_tick = self.__ticks[symbol]
        p = PositionExt(
            tsym=symbol,
            netqty=qty,
            prd=ProductType.BO,
            price=curr_tick.ltp,
            direction=direction,
            sl_target=(curr_tick.ltp - sl) if direction == BuyorSell.Buy else (curr_tick.ltp + sl),
            tp_target=(curr_tick.ltp + tp) if direction == BuyorSell.Buy else (curr_tick.ltp - tp),
            datetime=curr_tick.datetime
        )
        self._add_position(p)

    def orders(self):
        return self.__orders

    def notify_tick(self, tick: Tick):
        self.__ticks[tick.symbol] = tick

        keys = []
        for k in self.__positions.keys():
            if k.startswith(tick.symbol):
                keys.append(k)

        self._check_triggers(tick, keys)

    def _check_triggers(self, tick, keys):
        for k in keys:
            pos = self.__positions.get(k)
            self._check_triggers_for_bo(tick, pos)

    def _check_triggers_for_bo(self, tick: Tick, pos: Position):
        if pos.product_type == ProductType.BO:
            if pos.direction == BuyorSell.Buy:
                if (tick.ltp >= pos.tp_target) or (tick.ltp <= pos.sl_target):
                    o = Order(
                        entry_time=pos.datetime,
                        entry_price=pos.price,
                        sl=pos.sl_target,
                        tp=pos.tp_target, 
                        direction=pos.direction,
                        exit_price=tick.ltp,
                        exit_time=tick.datetime,
                        profit=pos.price - tick.ltp
                    )
                    self.__orders.append(o)
                    self.__positions.pop(self._get_position_key(pos))
        else:
            if pos.direction == BuyorSell.Sell:
                if (tick.ltp <= pos.tp_target) or (tick.ltp >= pos.sl_target):
                    o = Order(
                        entry_time=pos.datetime,
                        entry_price=pos.price,
                        sl=pos.sl_target,
                        tp=pos.tp_target, 
                        direction=pos.direction,
                        exit_price=tick.ltp,
                        exit_time=tick.datetime,
                        profit=tick.ltp - pos.price
                    )
                    self.__orders.append(o)
                    self.__positions.pop(self._get_position_key(pos))


class ShoonyaBroker(BaseBroker):
    _positions = None
    _limits = None

    def __init__(self, session: ShoonyaSession, polling_interval=1) -> None:
        self._session = session
        self._client = self._session.init_client()

        self._get_positions(polling_interval)

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
            if p is not None:
                self._positions = pd.DataFrame(p)
                self._positions.netqty = pd.to_numeric(self._positions.netqty)
            time.sleep(poll_interval)

    def positions(self, symbol: str, product_type: str=ProductType.BO):
        if self._positions is None:
            return None
        exch, sym = symbol.split(':')
        result = self._positions[
            (self._positions.tsym == sym) & 
            (self._positions.exch == exch) & 
            (self._positions.prd == product_type) & 
            (self._positions.netqty > 0)
        ]
        if not len(result.index):
            return None
        return [ Position.model_validate(i) for i in result.to_dict('records')]

    def orders(self):
        raise NotImplementedError

    def limits(self):
        return self._limits