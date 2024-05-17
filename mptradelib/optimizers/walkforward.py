import pydantic as pyd
import pandas as pd
from mptradelib.backtest import Backtest
from typing import Callable, List
from tqdm import tqdm

try:
    import plotly.graph_objects as go
except ImportError:
    pass

class Stage(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    stage_no: int
    stage_size: int
    test_fraction: float = 0.25

    @property
    def test_df(self):
        test_size = self._get_test_size()
        return self.stage_df[len(self.stage_df)-test_size : ]
    
    @property
    def opt_df(self):
        test_size = self._get_test_size()
        return self.stage_df[:len(self.stage_df)-test_size]
    
    @property
    def stage_df(self):
        test_size = self._get_test_size()
        start_idx = test_size * self.stage_no
        return self.df[start_idx:start_idx + self.stage_size]
    
    def _get_test_size(self):
        return int(self.stage_size * self.test_fraction)

    def optimize(self, compute, **params: dict):
        b = Backtest(self.opt_df, compute=compute)
        return b.optimize(params)


class Walkforward(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    stage_count: int = 6
    test_fraction: float = 0.25
    strategy: Callable[[pd.DataFrame, dict],pd.DataFrame]

    def _create_stages(self) -> Stage:
        stage_size = int(len(self.df)/(1 + (self.stage_count-1) * self.test_fraction))
        stages = []
        for i in range(self.stage_count):
            stage = Stage(df=self.df, stage_no=i, stage_size=stage_size, test_fraction=self.test_fraction)
            stages.append(stage)
        return stages

    def optimize(self, **params: dict):
        show_progressbar = params.pop('show_progressbar', True)

        results = []
        stages = self._create_stages()
        if show_progressbar:
            with tqdm(total=len(stages)) as pb:
                for stage in stages:
                    r = stage.optimize(self.strategy,**params)
                    results.append(r)
                    pb.update()
        else:
            for stage in stages:
                r = stage.optimize(self.strategy,**params)
                results.append(r)
        return results

    def _get_date(self, d, index):
        return d.datetime.iloc[index].strftime("%Y-%m-%d")

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

    dfs: List[pd.DataFrame]
    stage_count: int = 6
    test_fraction: float = 0.25
    strategy: Callable[[pd.DataFrame, dict],pd.DataFrame]

    def optimize(self, **params: dict):
        results = []
        with tqdm(total=len(self.dfs)) as pb:
            for df in self.dfs:
                w = Walkforward(
                    df=df, 
                    stage_count=self.stage_count, 
                    test_fraction=self.test_fraction, 
                    strategy=self.strategy
                )
                r = w.optimize(**params)
                results.append(r)
                pb.update()
        return self._aggregate(results)
    
    def _aggregate(self, results):
        stages = []
        for i in range(len(results[0])):
            stage = []
            for sym in results:
                stage.append(sym[i])
            stages.append(stage)

        final_result = []
        for stage in stages:
            pp = []
            for i in range(len(stage[0])):
                result_agg = {
                    'trades': []
                }
                for j in range(len(stage)):
                    result_agg['params'] = stage[j][i]['params']
                    result_agg['trades'].append(stage[j][i]['trades'])
                result_agg['trades'] = pd.concat(result_agg['trades'])
                pp.append(result_agg)
            final_result.append(pp)
        return final_result

        