import datetime as dt
import pandas as pd
import numpy as np
from typing import Callable
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope
from retry import retry
import multiprocessing as mp
from .broker.broker import Order
from typing import List, Union
from .utils import Tearsheet
from pydantic import BaseModel
from functools import partial


class urange(BaseModel):
    start: Union[int, float]
    stop: Union[int, float]
    step: Union[int, float] = 1

class Backtest:
    params: dict = {
        "sl": None,
        "tp": None
    }

    def __init__(self, 
                 data: pd.DataFrame, 
                 compute: Callable[[pd.DataFrame, dict], 
                 pd.DataFrame]=None, 
                 tp=2, sl=1,
                 intraday_exit_time: dt.time = dt.time(15,10,0),
                 intraday=True):
        self.data = data
        self.compute = compute
        self.params['tp'] = tp
        self.params['sl'] = sl
        self.intraday = intraday
        self.intraday_exit_time = intraday_exit_time

    def run(self, **kwargs):
        self.params.update(kwargs)

        data = self.data.copy()
        
        data = self.compute(data, self.params)
        
        if 'entries' not in data.columns.to_list():
            raise KeyError('column "entries" not found.')
        
        try:
            data.loc[data.entries != 0, 'entry_price'] = data.open.shift(-1)
        except Exception as e:
            print(data)
            raise e

        data.loc[data.entries == 1, 'sl'] = data.close * (1 - self.params['sl']/100)
        data.loc[data.entries == 1, 'tp'] = data.close * (1 + self.params['tp']/100)

        data.loc[data.entries == -1, 'sl'] = data.close * (1 + self.params['sl']/100)
        data.loc[data.entries == -1, 'tp'] = data.close * (1 - self.params['tp']/100)
        
        data['exit_price'] = data.open.shift(-1)
        data['trade_time'] = data.datetime.shift(-1)

        orders = self._generate_orders(data)
        out_df = pd.DataFrame([o.model_dump() for o in orders])
        return out_df, data

    def _exit_position(self, row, position, orders):
        position.exit_price = row.exit_price
        position.exit_time = row.trade_time
        position.set_profit()
        orders.append(position)
        position = None
        return position, orders

    def _in_intraday_window(self, row):
        return row.datetime.time() < self.intraday_exit_time

    def _generate_orders(self, data:pd.DataFrame):
        position = None
        orders = []
        for row in data.itertuples():
            if position is not None:
                if self.intraday and not self._in_intraday_window(row):
                    position, orders = self._exit_position(row, position, orders)
                
                elif position.direction == 1:
                    if row.close <= position.sl or row.close >= position.tp:
                        position, orders = self._exit_position(row, position, orders)
                        
                elif position.direction == -1:
                    if row.close >= position.sl or row.close <= position.tp:
                        position, orders = self._exit_position(row, position, orders)

            if row.entries != 0 and position is None and self._in_intraday_window(row):
                position = Order(entry_time=row.trade_time, entry_price=row.entry_price, sl=row.sl, tp=row.tp, direction=row.entries)
        return orders

    def _define_space(self, params):
        space = {}
        for k, v in params.items():
            if not isinstance(v, urange):
                raise TypeError(f"{v} is of type {type(v)}. Needs to be of type 'urange'.")
            
            if isinstance(v.step, int):
                space[k] = scope.int(hp.quniform(k, v.start, v.stop, v.step))
            elif isinstance(v.step, float):
                space[k] = hp.quniform(k, v.start, v.stop, v.step)
            else:
                raise ValueError(f"step of {k} can not be {type(v.step)}")
        return space

    @retry(tries=10)
    def optimize(self, kwargs, opt_param='profit'):
        show_progressbar = kwargs.pop('show_progressbar', False)
        max_evals = kwargs.pop('max_evals', 3000)
        space = self._define_space(kwargs)
        
        def objective(params):
            r = self.run(**params)
            if not len(r):
                return {'loss': 0, 'status': STATUS_OK}
            
            if opt_param == 'profit':
                loss = -np.sum(r[0].profit) if len(r[0]) else 0
                return {'loss': loss, 'status': STATUS_OK}
            elif opt_param == 'com':
                t = Tearsheet(r[0])
                ncom, com = t.fund_growth()
                return {'loss': -com, 'status': STATUS_OK}
            elif opt_param == 'winrate':
                t = Tearsheet(r[0])
                wr = t.win_rate()
                return {'loss': -wr, 'status': STATUS_OK}
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            show_progressbar=show_progressbar
        )
        p = {k: int(v) for k, v in best.items()}
        r = self.run(**p)
        return best, np.sum(r[0].profit)
        