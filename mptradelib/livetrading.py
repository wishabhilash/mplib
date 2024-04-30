from .feed import Datas, Tick, Series
from .broker.broker import BaseBroker

class BaseStrategy:
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def next(self, symbol: str, data: Series) -> None:
        raise NotImplementedError

class LiveTrading:
    _datas: Datas = None
    _s: BaseStrategy
    b: BaseBroker

    def __init__(self, s: BaseStrategy.__class__) -> None:
        self._s = s()

    def set_datas(self, d: Datas):
        self._datas = d

    def set_broker(self, b: BaseBroker):
        self.b = b

    def _next(self, t: Tick):
        self._s.next(t.symbol, self._datas[t.symbol])

    def run(self, **kwargs):
        if self._datas is None:
            raise ValueError("'datas' not found")
        
        self._s.set_params(**kwargs)
        
        self._datas.run("ticks", self._next)
        while True:
            pass
