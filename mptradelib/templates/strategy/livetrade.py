from mptradelib.livetrade import BaseStrategy, LiveTrading
from mptradelib.broker.session import FyersSession, ShoonyaSession
from mptradelib.feed import Datas
from mptradelib.broker.broker import Historical, ShoonyaBroker, ProductType
import pandas_ta as ta
import redis
import requests
import os
from retry import retry
import click

@click.group()
def cli():
    pass

class MyStrategy(BaseStrategy):
    ema_fast: int = 20
    ema_slow: int = 50
    sl: int = 1
    tp: int = 2

    def next(self, symbol, data):
        ema_fast = ta.ema(data.df.close, self.ema_fast)
        ema_slow = ta.ema(data.df.close, self.ema_slow)

        if self.b.positions(symbol, ProductType.Intraday):
            return

        if (ema_fast.iloc[-2] > ema_slow.iloc[-2]) and (ema_fast.iloc[-3] < ema_slow.iloc[-3]):
            self.b.buy(symbol, qty=10)


@retry(tries=10, delay=2)
def subscribe(symbols):
    subs = {
        "symbols": symbols
    }
    result = requests.post(os.getenv("SUBSCRIPTION_ENDPOINT"), data=subs)
    if result.status_code != 200:
        raise Exception("unable to subscribe to symbols")

@click.command()
# @click.option('--symbols', default=[], help="Ex. NSE:SBIN-EQ,NSE:CANB-EQ")
# @click.option('--timeframe', default=1, help="timeframe in minutes")
# @click.argument('run')
def run(symbols, timeframe):
    r = redis.Redis(
        host=os.getenv('REDIS_HOST'),
        port=os.getenv('REDIS_PORT'),
        decode_responses=True # <-- this will ensure that binary data is decoded
    )

    h = Historical(FyersSession())
    datas = Datas(r, h).resample(timeframe)
    datas.load(symbols=symbols)

    sb = ShoonyaBroker(ShoonyaSession())

    subscribe(symbols)
    
    l = LiveTrading(MyStrategy)
    l.set_datas(datas)
    l.set_broker(sb)
    l.run(ema_fast=20, ema_slow=50, ema_trend=200, sl=1, tp=2)
