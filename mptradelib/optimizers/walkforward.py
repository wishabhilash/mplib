import pydantic as pyd
import pandas as pd
from mptradelib.backtest import Backtest
from typing import Callable
from tqdm import tqdm
import plotly.graph_objects as go

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

    def run(self, compute, **params: dict):
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

    def run(self, **params: dict):
        results = []
        stages = self._create_stages()
        with tqdm(total=len(stages)) as pb:
            for stage in stages:
                r = stage.run(self.strategy,**params)
                results.append(r)
                pb.update()
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