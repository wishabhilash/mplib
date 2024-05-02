from .feed import Datas, Tick, Series
from .broker.broker import BaseBroker
import pydantic
import copy

class BaseStrategy(pydantic.BaseModel):
    def next(self, symbol: str, data: Series) -> None:
        raise NotImplementedError

class LiveTrading:
    _datas: Datas = None
    _s: BaseStrategy
    b: BaseBroker
    _p: dict = {}

    def __init__(self, s: BaseStrategy.__class__) -> None:
        self._s = s()

    def set_datas(self, d: Datas):
        self._datas = d

    def set_broker(self, b: BaseBroker):
        self.b = b

    def _next(self, t: Tick):
        params = self._p.get(t.symbol, {})
        for k, v in params.items():
            setattr(self._s, k, v)
        self._s.next(t.symbol, self._datas[t.symbol])

    def _process_params(self, kwargs):
        multisym_params_flag = True
        for symbol in self._datas.symbols():
            if symbol not in kwargs:
                multisym_params_flag = False
                break
        
        if multisym_params_flag:
            return kwargs
                
        default_params_flag = True
        if not multisym_params_flag:
            for v in kwargs.values():
                if not any([isinstance(v, t) for t in [int, float, str, bool]]):
                    default_params_flag = False

        if default_params_flag:
            return {symbol: copy.copy(kwargs) for symbol in self._datas.symbols()}

        if not multisym_params_flag and not default_params_flag:
            raise ValueError("invalid strategy params provided")

    def run(self, **kwargs):
        if self._datas is None:
            raise ValueError("'datas' not found")
        
        if not self.b:
            raise ValueError('broker not found')
        
        self._p = self._process_params(kwargs)

        self._datas.run("ticks", self._next)
        while True:
            pass
