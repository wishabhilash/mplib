import datetime as dt
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

position = None
orders = []

@dataclass
class Order:
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

class Backtest:
    params: dict = {
        "sl": None,
        "tp": None
    }

    __position: Order = None
    __orders = []

    def __init__(self, data: pd.DataFrame, compute: Callable[[pd.DataFrame, dict], pd.DataFrame]=None, tp=2, sl=1, intraday=True):
        self.data = data
        self.compute = compute
        self.params['tp'] = tp
        self.params['sl'] = sl
        self.intraday = intraday

    def run(self, **kwargs):
        self.params.update(kwargs)

        data = self.data.copy()
        
        data = self.compute(data, self.params)
        
        if 'entries' not in data.columns.to_list():
            raise KeyError('column "entries" not found.')
        
        data.loc[data.entries != 0, 'entry_price'] = data.open.shift(-1)

        data.loc[data.entries == 1, 'sl'] = data.close * (1 - self.params['sl']/100)
        data.loc[data.entries == 1, 'tp'] = data.close * (1 + self.params['tp']/100)

        data.loc[data.entries == -1, 'sl'] = data.close * (1 + self.params['sl']/100)
        data.loc[data.entries == -1, 'tp'] = data.close * (1 - self.params['tp']/100)
        
        data['exit_price'] = data.open.shift(-1)
        data['trade_time'] = data.datetime.shift(-1)

        for row in data.itertuples():
            self._backtest(row)
        out_df = pd.DataFrame([asdict(o) for o in self.__orders])
        self.__orders = []
        self.__position = None
        return out_df, data

    def _exit_position(self, row):
        self.__position.exit_price = row.exit_price
        self.__position.exit_time = row.trade_time
        self.__position.set_profit()
        self.__orders.append(self.__position)
        self.__position = None

    def _in_intraday_window(self, row):
        return row.datetime.time() <= dt.time(15,10,0)

    def _backtest(self, row):
        if self.__position is not None:
            
            if self.intraday and not self._in_intraday_window(row):
                self._exit_position(row)
            
            elif self.__position.direction == 1:
                if row.close <= self.__position.sl or row.close >= self.__position.tp:
                    self._exit_position(row)
                    
            elif self.__position.direction == -1:
                if row.close >= self.__position.sl or row.close <= self.__position.tp:
                    self._exit_position(row)

        if row.entries != 0 and self.__position is None and self._in_intraday_window(row):
            self.__position = Order(entry_time=row.trade_time, entry_price=row.entry_price, sl=row.sl, tp=row.tp, direction=row.entries)

    def _optimizer(self, **kwargs):
        space = { k: scope.int(hp.quniform(k, v.start, v.stop, v.step)) for k, v in kwargs.items()}
        
        def objective(params):
            r = self.run(**params)
            return {'loss': -np.sum(r[0].profit), 'status': STATUS_OK}
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
        )
        p = {k: int(v) for k, v in best.items()}
        r = self.run(**p)
        return best, np.sum(r[0].profit)
    
    def optimize(self, runs=5, **kwargs):
        results = []
        for _ in tqdm(range(runs)):
            r = self._optimizer(**kwargs)
            results.append(r)

        max_profit = 0
        result = None
        for r in results:
            if r[1] > max_profit:
                max_profit = r[1]
                result = r
        return result
