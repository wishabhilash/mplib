import datetime as dt
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket.data_ws import FyersDataSocket
import time
import random
import threading
import queue

from redis import Redis

try:
    import redis
except ImportError:
    pass

import json


class BaseTicker:
    _r: redis.Redis = None
    _is_live: bool = False

    def __init__(self, broker: None, r: redis.Redis, channel="default-ticks") -> None:
        self._broker = broker
        self._r = r
        self._channel = channel

    def run(self):
        raise NotImplementedError
    
    def subscribe(self, symbols: list):
        pass

    @property
    def is_live(self):
        return self._is_live

class LiveTicker(BaseTicker):
    _client: fyersModel.FyersModel = None
    _ws: FyersDataSocket = None

    def __init__(self, broker: None, r: Redis, channel="default-ticks") -> None:
        super().__init__(broker, r, channel)
        self._q = queue.Queue(1)

    def run(self):
        if self._client is None and self._broker is not None:
            self._client = self._broker.init_client()
        
        self._ws = FyersDataSocket(
            access_token=self._client.token,
            litemode=False,
            on_connect=self.__on_connect,
            on_message=self.__on_message,
            on_error=self.__on_error,
            on_close=self.__on_close
        )

        self._ws.connect()

        while True:
            syms = self._q.get()
            if syms:
                self._ws.subscribe(syms)

    def __on_connect(self):
        print("connected")
        self._is_live = True

    def subscribe(self, symbols: list):
        self._q.put(symbols)

    def __on_message(self, msg):
        self._r.publish(self._channel, json.dumps(msg))

    def __on_error(self, err):
        print(err)

    def __on_close(self):
        print("close")


def round_nearest(x, a):
    return round(round(x / a) * a, 2)
        

class MockTicker(BaseTicker):
    _subscriptions: set = set()

    def __feed_worker(self):
        while True:
            for s in self._subscriptions:
                price = round_nearest(random.uniform(100.0, 500.0), 0.05)
                d = {'ltp': price, 'symbol': s, 'type': 'sf', 'last_traded_time': int(dt.datetime.now().timestamp())}
                time.sleep(round(random.uniform(0.0, 1.0), 2))
                self._r.publish(self._channel, json.dumps(d))

    def run(self):
        self._is_live = True
        t = threading.Thread(target=self.__feed_worker)
        t.start()
        
    def subscribe(self, symbols: list):
        if self.is_live:
            for s in symbols:
                self._subscriptions.add(s)
