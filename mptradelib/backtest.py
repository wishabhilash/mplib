import datetime as dt
import pandas as pd
from typing import Callable
from retry import retry
from .broker.broker import Order
from itertools import product
import multiprocessing as mp


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
            print(data, '.............')
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
        opt_params = {}
        constant_params = {}
        for k, v in params.items():
            if isinstance(v, range):
                opt_params[k] = v
            else:
                constant_params[k] = v
        return opt_params, constant_params
    
    def _get_overfitting_score(self, t):
        t['abs_profit'] = t.profit.abs()
        ts = t.sort_values(by=['abs_profit'], ascending=False)
        os = (ts[:int(len(ts)/100)].abs_profit.sum()/ts.abs_profit.sum())*100
        rationalized_df = ts[int(len(ts)/100):]
        return os, rationalized_df

    def _objective(self, params):
        r, _ = self.run(**params)
        os, rdf = self._get_overfitting_score(r.copy())
        if os > 10:
            r = rdf
            print("removing event effect - ", params)
        return {'params': params, 'trades': r}


    @retry(tries=10)
    def optimize(self, kwargs):
        opt_params, constant_params = self._define_space(kwargs)
        
        results = []
        params = []
        for p in product(*opt_params.values()):
            param = dict(zip(opt_params.keys(), p))
            param.update(constant_params)
            params.append(param)
            
        with mp.Pool(mp.cpu_count()) as p:
            results = p.map(self._objective, params)
        return results