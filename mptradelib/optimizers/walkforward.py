import pydantic as pyd
import pandas as pd
from mptradelib.backtest import Backtest
from typing import Callable, List, Dict
from tqdm import tqdm
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as e:
    print('install plotly by "pip install plotly"')


class StrategyFuncs(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    compute: Callable[[pd.DataFrame, dict], pd.DataFrame]
    exit_func: Callable[[pd.DataFrame, dict], None] = None

class Stage(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    main_df: pd.DataFrame = pyd.Field(..., alias='df')
    stage_no: int
    stage_size: int
    test_fraction: float = 0.25
    _opt_result: pd.DataFrame = None

    @property
    def test_df(self):
        test_size = self._get_test_size()
        return self.df[len(self.df)-test_size : ]
    
    @property
    def opt_df(self):
        test_size = self._get_test_size()
        return self.df[:len(self.df)-test_size]
    
    @property
    def df(self):
        test_size = self._get_test_size()
        start_idx = test_size * self.stage_no
        return self.main_df[start_idx:start_idx + self.stage_size]
    
    @property
    def opt_result(self):
        return self._opt_result
    
    def _get_test_size(self):
        return int(self.stage_size * self.test_fraction)

    def optimize(self, strategy_funcs: StrategyFuncs, **params: dict):
        b = Backtest(self.opt_df, compute=strategy_funcs.compute, exit_func=strategy_funcs.exit_func)
        self._opt_result = b.optimize(params)
        return self

    def validate(self, strategy_funcs: StrategyFuncs, **kwargs):
        b = Backtest(self.test_df, compute=strategy_funcs.compute, exit_func=strategy_funcs.exit_func)
        trades, _ = b.run(**kwargs)
        return trades


class Walkforward(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    stage_count: int = 6
    test_fraction: float = 0.25
    strategy_funcs: StrategyFuncs
    stages: List[Stage] = []

    def _create_stages(self) -> List[Stage]:
        stage_size = int(len(self.df)/(1 + (self.stage_count-1) * self.test_fraction))
        stages = []
        for i in range(self.stage_count):
            stage = Stage(df=self.df, stage_no=i, stage_size=stage_size, test_fraction=self.test_fraction)
            stages.append(stage)
        return stages

    def optimize(self, **params: dict):
        show_progressbar = params.pop('show_progressbar', True)

        stages: Stage = self._create_stages()
        if show_progressbar:
            with tqdm(total=len(stages)) as pb:
                for stage in stages:
                    s = stage.optimize(self.strategy_funcs, **params)
                    self.stages.append(s)
                    pb.update()
        else:
            for stage in stages:
                stage.optimize(self.strategy_funcs, **params)
                self.stages.append(stage)
        return self
    
    @property
    def opt_result(self):
        return [s.opt_result for s in self.stages]

    def _get_date(self, d, index):
        return d.datetime.iloc[index].strftime("%Y-%m-%d")
    
    def validate(self, strategy_func, params):
        results = []
        for i in range(len(self.stages)):
            r = self.stages[i].validate(strategy_func, **params[i])
            results.append(r)
        return pd.concat(results)

    def plot_splits(self):
        stages = self._create_stages()
        fig = go.Figure()
        for i in range(len(stages)):
            s = stages[i]
            opt_set, test_set = s.opt_df, s.test_df
            fig.add_trace(go.Scatter(
                name='Optimization', 
                y=[-i,-i],
                x=[self._get_date(opt_set, 0), self._get_date(opt_set, len(opt_set) - 1)],
                mode="lines",
                line=dict(color="blue",width=10)
            ))
            fig.add_trace(go.Scatter(
                name='Test',
                y=[-i,-i],
                x=[self._get_date(test_set, 0), self._get_date(test_set, len(test_set) - 1)],
                mode="lines",
                line=dict(color="red",width=10)
            ))


        fig.update_layout(
            yaxis_range=[-len(stages)-1, 1],
            hovermode='x unified',
            showlegend=False,
            title="Walkforward stages",
        )
        fig.update_yaxes(visible=False)
        fig.show()

class MultiSymbolWalkforwardAnalysis(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    dfs: Dict[str, pd.DataFrame]
    stage_count: int = 6
    test_fraction: float = 0.25
    strategy_funcs: StrategyFuncs
    walkforwards: Dict[str, Walkforward] = {}
    __computed_result = None

    def optimize(self, **params: dict):
        with tqdm(total=len(self.dfs)) as pb:
            for k, df in self.dfs.items():
                w = Walkforward(
                    df=df, 
                    stage_count=self.stage_count, 
                    test_fraction=self.test_fraction, 
                    strategy_funcs=self.strategy_funcs
                )
                w.optimize(**params)
                pb.update()
                self.walkforwards[k] = w
        return self
    
    @property
    def opt_result(self):
        return [w.opt_result for w in self.walkforwards.values()]

    def transform_to_agg_stages(self):
        if self.__computed_result is not None:
            return self.__computed_result
        
        stages = []
        for i in range(len(list(self.walkforwards.values())[0].opt_result)):
            stage = []
            for sym in self.walkforwards.values():
                stage.append(sym.stages[i])
            stages.append(stage)

        final_result = []
        for stage in stages:
            pp = []
            for i in range(len(stage[0].opt_result)):
                result_agg = {
                    'trades': []
                }
                for j in range(len(stage)):
                    result_agg['params'] = stage[j].opt_result[i]['params']
                    result_agg['trades'].append(stage[j].opt_result[i]['trades'])
                result_agg['trades'] = pd.concat(result_agg['trades'])
                pp.append(result_agg)
            final_result.append(pp)
        self.__computed_result = final_result
        return self.__computed_result

    def create_pre_pivot_dfs(self, param_cols: tuple):
        final_result = []
        results_agg = []
        default_cols = ['profit', 'trades', 'profit_perc']
        cols = param_cols + default_cols
        for i in range(len(list(self.walkforwards.values())[0].stages)):
            for stage in self.transform_to_agg_stages()[i]:
                if stage['trades'].empty:
                    continue
                row = [stage['params'][p] for p in param_cols]
                row.append(round(stage['trades'].profit.sum(), 2))
                row.append(stage['trades'])
                row.append(self.profit_perc_sum(stage['trades']))
                results_agg.append(row)
            final_result.append(pd.DataFrame(results_agg, columns=cols))
        return final_result

    def find_optimal_params(self, cols: tuple):
        extracted_params = []
        ppdf = self.create_pre_pivot_dfs(cols)
        for d in ppdf:
            profit_perc = d.pivot_table(values='profit_perc', index=[cols[0]], columns=[cols[1]])
            p = self._find_optimal_param_for_test(profit_perc)
            extracted_params.append(dict(zip(cols, p)))
        return extracted_params
    
    def validate(self, sym, strategy_func, params):
        w = self.walkforwards[sym]
        return w.validate(strategy_func, params)
        
    def _find_optimal_param_for_test(self, d):        
        sample = d.copy()
        sample['mean'] = sample.mean(axis=1)
        sample['sdv'] = sample.std(axis=1)
        scoredf = sample

        scoredf = scoredf.sort_values('mean')
        scoredf['mean_score'] = np.arange(len(scoredf))
        scoredf = scoredf.sort_values('sdv', ascending=False)
        scoredf['sdv_score'] = np.arange(len(scoredf))
        scoredf = scoredf.sort_index()
        scoredf['score'] = scoredf.mean_score * scoredf.sdv_score

        selected_param1 = scoredf.sort_values('score').score.idxmax()
        selected_row = d.loc[selected_param1]

        selected_param2 = selected_row.index[int(len(selected_row)/2)]
        return selected_param1, selected_param2

    def profit_perc_sum(self, d):
        perc = d.profit/d.entry_price * 100
        return perc.sum()
        