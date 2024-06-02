import pandas as pd
import numpy as np
from .simulated_trades import SimulateTrades
import datetime as dt
from .chart import *
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("please install plotly by 'pip install plotly' to enable plotting")


class Tearsheet:
    chart_fig: go.Figure = None

    def __init__(self, s: SimulateTrades) -> None:
        self._s = s
        self.mdf = self._s.run()

    def profit_factor(self):
        r = self.mdf[self.mdf.sim_profit > 0].sim_profit.sum()/abs(self.mdf[self.mdf.sim_profit < 0].sim_profit.sum())
        return round(r, 2)

    def total_profit(self):
        return round(np.sum(self.mdf.sim_profit), 2)
    
    def peak_profit(self):
        return round(self.mdf.sim_profit.cumsum().max(), 2)
    
    def win_vs_loss(self):
        return (
            len(self.mdf[self.mdf.sim_profit >= 0]),
            len(self.mdf[self.mdf.sim_profit < 0]),
        )
    
    def win_rate(self):
        r = self.win_vs_loss()
        return round(r[0]/sum(r) * 100, 2)
    
    def drawdown(self):
        if self.mdf.empty:
            return None, None
        cum_profit = self.mdf.sim_profit.cumsum()
        curr_max = cum_profit.expanding().max()
        dd = cum_profit - curr_max
        perc = dd/curr_max * 100
        return dd, perc

    def max_drawdown(self):
        dd, ddp = self.drawdown()
        if dd is None:
            return 0, 0
        return round(dd.min(), 2), round(ddp.min(), 2)
    
    def sharpe_ratio(self):
        n = len(self.mdf.sim_profit)
        return round(np.sqrt(n) * np.mean(self.mdf.sim_profit)/np.std(self.mdf.sim_profit), 2)
    
    def avg_profit(self):
        return self.mdf[self.mdf.sim_profit > 0].sim_profit.mean()
    
    def avg_loss(self):
        return self.mdf[self.mdf.sim_profit < 0].sim_profit.mean()
    
    def risk_reward_ratio(self):
        return abs(self.avg_profit()/self.avg_loss())
    
    def long_short_composition(self):
        positive = self.mdf[self.mdf.sim_profit > 0].groupby('direction').agg({'sim_profit': 'sum'})
        negative = self.mdf[self.mdf.sim_profit < 0].groupby('direction').agg({'sim_profit': 'sum'}).rename(columns={'sim_profit': 'loss'})
        return pd.concat([positive, negative], axis=1).round(2).rename(index={1: 'long', -1: 'short'})
    
    def final_balance(self):        
        return round(self.mdf.sim_profit.sum() + self._s._initial_cash, 2)

    def print(self):
        mdd, _ = self.max_drawdown()
        output = f'''
Performance metrics (From: {self.mdf.iloc[0].entry_time.date()} To: {self.mdf.iloc[-1].entry_time.date()})

Initial balance:        {self._s._initial_cash}
Final balance:          {self.final_balance()}
Total profit:           {self.total_profit()}
Peak profit:            {self.peak_profit()}
Profit factor:          {self.profit_factor()}
Win/Loss:               {self.win_vs_loss()[0]}/{self.win_vs_loss()[1]}
Win Rate:               {self.win_rate()} %
Avg. Profit:            {self.avg_profit()}
Avg. Loss:              {self.avg_loss()}
Risk-Reward Ratio:      {self.risk_reward_ratio()}
Max Drawdown:           {mdd}
Sharpe ratio:           {self.sharpe_ratio()}
'''
        print(output)

    def plot(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Cum. profit", "Daywise", "Long Short Split", "Hourwise"))
        self.plot_equity_curve(fig)
        self.plot_daywise_metrics(fig)
        self.plot_long_short_composition(fig)
        self.plot_hourwise_metrics(fig)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            showlegend=True
        )
        fig.show()

    def plot_hourwise_metrics(self, fig: go.Figure):
        r = self.mdf.groupby(self.mdf.entry_time.dt.hour).agg({'sim_profit': 'sum'})
        gdf = self.mdf[self.mdf.sim_profit > 0].groupby(self.mdf.entry_time.dt.hour).agg({'sim_profit': 'count'})
        ldf = self.mdf[self.mdf.sim_profit < 0].groupby(self.mdf.entry_time.dt.hour).agg({'sim_profit': 'count'})

        trace = go.Bar(
            name='Total profit',
            x=r.index,
            y=r.sim_profit
        )
        fig.add_trace(trace, row=2, col=2)

        trace2 = go.Bar(
            name='Profit trades',
            x=gdf.index,
            y=gdf.sim_profit
        )
        fig.add_trace(trace2, row=2, col=2)

        trace3 = go.Bar(
            name='Loss trades',
            x=ldf.index,
            y=ldf.sim_profit
        )
        fig.add_trace(trace3, row=2, col=2)

    def plot_long_short_composition(self, fig: go.Figure):
        r = self.long_short_composition()
        trace = go.Bar(
            name='Profit',
            x=r.index,
            y=r.sim_profit,
            text=r.sim_profit,
            textposition="inside",
            textfont_color="white",
            marker_color='rgb(153, 204, 0, 120)'
        )
        fig.add_trace(trace, row=2, col=1)

        trace = go.Bar(
            name='Loss',
            x=r.index,
            y=r.loss,
            text=r.loss,
            textposition="inside",
            textfont_color="white",
            marker_color='red'
        )
        fig.add_trace(trace, row=2, col=1)

    def plot_daywise_metrics(self, fig: go.Figure):
        cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        r = self.mdf.groupby(self.mdf.entry_time.dt.day_name()).agg({'sim_profit': 'sum'}).reindex(cats)
        gdf = self.mdf[self.mdf.sim_profit > 0].groupby(self.mdf.entry_time.dt.day_name()).agg({'sim_profit': 'count'})
        ldf = self.mdf[self.mdf.sim_profit < 0].groupby(self.mdf.entry_time.dt.day_name()).agg({'sim_profit': 'count'})

        trace = go.Bar(
            name='Total profit',
            x=r.index,
            y=r.sim_profit
        )
        fig.add_trace(trace, row=1, col=2)

        trace2 = go.Bar(
            name='Profit trades',
            x=gdf.index,
            y=gdf.sim_profit
        )
        fig.add_trace(trace2, row=1, col=2)

        trace3 = go.Bar(
            name='Loss trades',
            x=ldf.index,
            y=ldf.sim_profit
        )
        fig.add_trace(trace3, row=1, col=2)
        
    def plot_equity_curve(self, fig: go.Figure):
        cum_profit = go.Scatter(
            name="Cumulative Profit",
            x=self.mdf.entry_time, 
            y=self.mdf.sim_profit.cumsum() + self._s._initial_cash,
            fill="tozeroy", 
            fillcolor="rgb(153, 204, 0, 120)", 
            line={'color': 'rgb(153, 204, 0, 120)'}, 
        )
        fig.add_trace(cum_profit, row=1, col=1)

        dd, _ = self.drawdown()
        drawdowns = go.Scatter(
            name="Drawdown",
            x=self.mdf.entry_time,
            y=dd,
            fill="tozeroy",
            fillcolor="rgba(228,128,68,120)",
            line={'color': 'rgba(228,128,68,120)'},
        )
        fig.add_trace(drawdowns, row=1, col=1)

    def _get_holidays(self, dates):
        if not len(dates):
            return []
        start = dates.head(1).iloc[0].date()
        end = dates.tail(1).iloc[0].date()
        date_range = pd.date_range(start, end).difference(dates)
        return list(date_range.strftime('%Y-%m-%d'))

    def _get_row_heights(self, charts):
        row_height = round(1/(len(charts) - 1 + 4), 2)
        row_heights = [ row_height for _ in range(len(charts) - 1)]
        return [4 * row_height] + row_heights

    def plot_chart(self, name, df, trades, 
                   from_date = dt.datetime.now().date(), 
                   to_date = dt.datetime.now().date() + dt.timedelta(days=1),
                   charts: List[Chart] = [], jupyter=False):
        
        if self.chart_fig is None:
            fig = make_subplots(
                rows=len(charts),
                cols=1, 
                subplot_titles=tuple([name] + [t.name for t in charts]),
                row_heights=self._get_row_heights(charts),
                vertical_spacing=0.02,
                shared_xaxes=True,
            )
            self.chart_fig = go.FigureWidget(fig) if jupyter is True else fig
        else:
            self.chart_fig.data = []
            self.chart_fig.layout['annotations'] = None
        
        row_count = len(charts)
        filtered_df = df[(df.datetime.dt.date >= from_date) & (df.datetime.dt.date < to_date)]
        filtered_trades = trades[(trades.entry_time.dt.date >= from_date) & (trades.entry_time.dt.date <= to_date)]

        cs_chart = go.Candlestick(
            x=filtered_df.datetime,
            open=filtered_df.open,
            high=filtered_df.high,
            low=filtered_df.low,
            close=filtered_df.close,
            showlegend=False,
            name=name,
        )

        self.chart_fig.add_trace(cs_chart, row=1, col=1)
        for i in range(len(charts)):
            chart = charts[i]
            for c in chart.children:
                if isinstance(c, Line):
                    self.chart_fig.add_trace(
                        c.get_trace(filtered_df), 
                        row=1 if chart.main else i + 1, 
                        col=1
                    )
                elif isinstance(c, Rect):
                    self.chart_fig.add_shape(
                        c.get_trace(filtered_df),
                        row=1 if chart.main else i + 1, 
                        col=1
                    )
                elif isinstance(c, Background):
                    self.chart_fig.add_vrect(
                        **c.get_trace(filtered_df), 
                        row=1 if chart.main else i + 1, 
                        col=1
                    )

        for row in filtered_trades.itertuples():
            self.chart_fig.add_annotation(x=row.entry_time, y=row.entry_price,
                text= "LE" if row.direction > 0 else "SE",
                showarrow=True,
                arrowhead=1,
                bgcolor="#ff7f0e",
            )
            self.chart_fig.add_annotation(x=row.exit_time, y=row.exit_price,
                text= "LEx" if row.direction > 0 else "SEx",
                showarrow=True,
                arrowhead=1,
                bgcolor="#ff7f0e",
            )

        self.chart_fig.update_layout(
            hoversubplots="axis",
            hovermode="x unified",
            grid=dict(rows=row_count, columns=1),
            autosize=True,
            margin=dict(l=30, r=30, t=40, b=30),
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        holidays = self._get_holidays(filtered_df.datetime)

        self.chart_fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[15.5, 9.25], pattern="hour"),
            ]
        )

        return self.chart_fig