import json
import pandas as pd
from tqdm import tqdm
import os
import plotly.graph_objects as go
from ..utils import Tearsheet
from ..backtest import Backtest
from typing import Callable


class WalkForward:
    def __init__(self, df: pd.DataFrame, compute: Callable[[pd.DataFrame, dict], None], optimization_params: dict, freq='W', cache_path = ".") -> None:
        self.df = df
        self._compute = compute
        self._oparam = optimization_params
        self._freq = freq
        self._cache_path = os.path.join(cache_path, 'walkforward-reports')
        if not os.path.exists(self._cache_path):
            os.mkdir(self._cache_path)

    def _get_date(self, d, index):
        return d.datetime.iloc[index].strftime("%Y-%m-%d")

    def _generate_date_ranges(self, df):
        rng = pd.date_range(self._get_date(df, 0), self._get_date(df, len(df) - 1), freq=self._freq)
        rng = pd.Series(rng)
        return pd.DataFrame({
            'mstart': rng,
            'mend': rng.shift(-1),
        }).dropna()

    def data_splitter(self, train_size, test_size, step=1):
        date_ranges = self._generate_date_ranges(self.df)
        
        splits = []
        window = (train_size + test_size) * step
        for dfs in date_ranges.rolling(window, step=step):
            if len(dfs.index) < window:
                continue

            s = dfs.iloc[0].mstart.date().strftime("%Y-%m-%d")
            e = dfs.iloc[window - 1 - (test_size * step)].mend.date().strftime("%Y-%m-%d")
            e2 = dfs.iloc[window - 1].mend.date().strftime("%Y-%m-%d")
            opt_set = self.df.loc[s:e]
            test_set = self.df.loc[e:e2]
            splits.append((opt_set, test_set, self._cache_path))
        return splits
    
    def _calculate_result(self):
        dflist = []
        for dirname in os.listdir(self._cache_path):
            if dirname == '.DS_Store':
                continue
            try:
                filepath = os.path.join(self._cache_path, dirname, "report.csv")
            except Exception as e:
                print(f'reading {filepath}')
                raise e
            dff = pd.read_csv(filepath)
            dflist.append(dff)

        fdf = pd.concat(dflist)
        fdf = fdf.sort_values(by=['entry_time']).drop_duplicates().reset_index(drop='index')
        fdf.entry_time = pd.to_datetime(fdf.entry_time)
        fdf.exit_time = pd.to_datetime(fdf.exit_time)
        fdf.to_csv('result.csv', index=False)

        t = Tearsheet(fdf)
        t.print()
        t.plot()

    def _optimize(self, df, runs=5, show_progress=False):
        b = Backtest(df, self._compute, sl=1, tp=2, intraday=True)
        r = b.optimize(
            runs=runs,
            show_progress=show_progress,
            **self._oparam
        )
        return r[0]

    def _create_optimization_worker(self, params, runs=5):
        dfopt, dftest, reports_path = params

        filename = f'{dftest.iloc[0].datetime.strftime("%Y-%m-%d")}-{dftest.iloc[-1].datetime.strftime("%Y-%m-%d")}'
        dirpath = os.path.join(reports_path, filename)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        param_path = os.path.join(dirpath, 'params.json')
        report_path = os.path.join(dirpath, 'report.csv')

        if os.path.exists(report_path):
            return pd.read_csv(report_path)

        r = self._optimize(dfopt, runs)
        with open(param_path, 'w') as f:
            f.write(json.dumps(r))

        c = Backtest(dftest, self._compute, sl=1, tp=2, intraday=True)
        out = c.run(**{k: int(v) for k, v in r.items()})
        out[0].to_csv(report_path, index=False)
        return out[0]


    def run(self, train_size, test_size, step=1, runs=5):
        splits = self.data_splitter(train_size, test_size, step)
        self._plot_splits(splits)
        self._optimize_splits(splits, runs)
        self._calculate_result()

    def _optimize_splits(self, splits, runs):
        with tqdm(total=len(splits)) as pbar:
            for p in splits:
                self._create_optimization_worker(p, runs)
                pbar.update()

    def _plot_splits(self, splits):
        fig = go.Figure()
        for i in range(len(splits)):
            s = splits[i]
            opt_set, test_set, _ = s
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
            yaxis_range=[-len(splits)-1, 1],
            hovermode='x unified',
            showlegend=False,
            title="Walkforward split",
        )
        fig.update_yaxes(visible=False)
        fig.show()

