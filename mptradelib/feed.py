import copy
import redis
import json
import threading
import datetime as dt
import pandas as pd
from pydantic import BaseModel, Field
from collections import OrderedDict
from typing import Callable, List
from .broker.broker import Historical, Tick
from retry import retry
import requests
import os


class Candle(BaseModel):
    symbol: str = Field(None, exclude=True, repr=False)
    datetime: dt.datetime
    ltp: float = Field(None, exclude=True, repr=False)
    open: float = None
    high: float = None
    low: float = None
    close: float = None

    class Config:
        fields = {'value': {'exclude': True}}

    def update(self, tick: Tick):
        self.open = tick.ltp if self.open is None else self.open
        self.high = tick.ltp if (self.high is None) or (tick.ltp > self.high) else self.high
        self.low = tick.ltp if (self.low is None) or (tick.ltp < self.low) else self.low
        self.close = tick.ltp
        return self

class Series(list):
    def __getitem__(self, *args, **kwargs) -> Candle:
        item = super().__getitem__(*args, **kwargs)
        return Candle.model_validate(item)

    @property
    def df(self):
        return pd.DataFrame(list(self))

class Datas:
    _datas: dict = {}
    _resample_period = 1

    def __init__(self, syms: List[str], r: redis.Redis, h: Historical, live: True) -> None:
        self._r = r
        self._h = h
        self.subscribe(syms)
        self._syms_map = dict(zip(syms, [True] * len(syms)))
        self.__live = live
    
    @property
    def is_live(self):
        return self.__live

    @retry(tries=10, delay=2)
    def subscribe(self, symbols):
        payload = json.dumps({
            "symbols": symbols
        })

        headers = {
            'Content-Type': 'application/json',
            'Accept':'application/json'
        }
        result = requests.post(os.getenv("SUBSCRIPTION_ENDPOINT"), data=payload, headers=headers)
        if result.status_code != 200:
            raise Exception("unable to subscribe to symbols")

    def resample(self, period=1):
        self._resample_period = period
        return self

    def _generate_key(self, a: dt.datetime):
        b = copy.copy(a)
        c = b - dt.timedelta(minutes=(b.minute % self._resample_period))
        return f"{c:%Y-%m-%d-%H-%M}"
    
    def symbols(self):
        return self._datas.keys()

    def update(self, tick: Tick):
        key = self._generate_key(tick.datetime)

        try:
            series = self._datas[tick.symbol]
        except KeyError:
            self._datas[tick.symbol] = OrderedDict()
            series = self._datas[tick.symbol]

        try:
            series[key] = Candle.model_validate(series[key]).update(tick).model_dump()
        except Exception as e:
            series[key] = Candle(
                datetime=tick.datetime.replace(second=0),
                symbol=tick.symbol,
                ltp=tick.ltp
            ).update(tick).model_dump()

    def _load_worker(self, symbol: str, start_date: dt.datetime, end_date: dt.datetime):
        try:
            series = self._datas[symbol]
        except KeyError:
            self._datas[symbol] = OrderedDict()
            series = self._datas[symbol]

        data = self._h.historical(symbol, self._resample_period, start_date, end_date)
        for d in data.to_dict('records'):
            c = Candle.model_validate(d)
            key = self._generate_key(c.datetime)
            series[key] = c.model_dump()

    def load(self, period=7):
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=period)

        threads = []
        for symbol in self._syms_map.keys():    
            t = threading.Thread(target=self._load_worker, args=[symbol, start_date, end_date])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
                
    def __getitem__(self, index):
        return Series(self._datas[index].values())
    
    def _listen(self, channel, on_message: Callable[[Tick],None]):
        con = self._r.pubsub()
        con.subscribe(channel)
        for msg in con.listen():
            try:
                d = json.loads(msg['data'])
                if not self._syms_map.get(d['symbol'], False):
                    continue

                tick = Tick(
                    datetime=dt.datetime.fromtimestamp(d['last_traded_time']),
                    ltp = d['ltp'],
                    symbol=d['symbol']
                )
                self.update(tick)
                on_message(tick)
            except Exception as e:
                print(e)

    def run(self, channel, on_message: Callable[[Tick],None]):
        if self.is_live:
            t = threading.Thread(target=self._listen, args=[channel, on_message])
            t.start()
        else:
            symbol = list(self._datas.keys())[0]
            series = self._datas.pop(symbol).values()
            for c in series:
                key = self._generate_key(c['datetime'])
                try:
                    self._datas[symbol][key] = c
                except KeyError:
                    self._datas[symbol] = OrderedDict()
                    self._datas[symbol][key] = c
                on_message(Tick(
                    datetime=c['datetime'],
                    ltp=c['close'],
                    symbol=symbol
                ))
