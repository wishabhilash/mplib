import pydantic as pyd
import pandas as pd
from mptradelib.backtest import Backtest
from typing import Callable, List
from tqdm import tqdm
import plotly.graph_objects as go

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

    def optimize(self, compute, **params: dict):
        b = Backtest(self.opt_df, compute=compute)
        self._opt_result = b.optimize(params)
        return self

    def validate(self):
        pass


class Walkforward(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    stage_count: int = 6
    test_fraction: float = 0.25
    strategy: Callable[[pd.DataFrame, dict],pd.DataFrame]
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
                    s = stage.optimize(self.strategy, **params)
                    self.stages.append(s)
                    pb.update()
        else:
            for stage in stages:
                stage.optimize(self.strategy, **params)
                self.stages.append(stage)
        return self
    
    @property
    def opt_result(self):
        return [s.opt_result for s in self.stages]

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
    walkforwards: List[Walkforward] = []
    __computed_result = None

    def optimize(self, **params: dict):
        with tqdm(total=len(self.dfs)) as pb:
            for df in self.dfs:
                w = Walkforward(
                    df=df, 
                    stage_count=self.stage_count, 
                    test_fraction=self.test_fraction, 
                    strategy=self.strategy
                )
                w.optimize(**params)
                pb.update()
                self.walkforwards.append(w)
        return self
    
    @property
    def opt_result(self):
        return [w.opt_result for w in self.walkforwards]

    def transform_to_agg_stages(self):
        if self.__computed_result is not None:
            return self.__computed_result
        
        stages = []
        for i in range(len(self.walkforwards[0].opt_result)):
            stage = []
            for sym in self.walkforwards:
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
        return final_result

        