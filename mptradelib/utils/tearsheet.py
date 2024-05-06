import pandas as pd
import numpy as np
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("please install plotly by 'pip install plotly' to enable plotting")


class Tearsheet:
    def __init__(self, result: pd.DataFrame, seed=10000) -> None:
        self.df = result
        self._seed = seed

    def total_profit(self):
        return round(np.sum(self.df.profit), 2)
    
    def peak_profit(self):
        return round(self.df.profit.cumsum().max(), 2)
    
    def win_vs_loss(self):
        return (
            len(self.df[self.df.profit >= 0]),
            len(self.df[self.df.profit < 0]),
        )
    
    def win_rate(self):
        r = self.win_vs_loss()
        return round(r[0]/sum(r) * 100, 2)
    
    def drawdown(self):
        cum_profit = self.df.profit.cumsum()
        curr_max = cum_profit.expanding().max()
        dd = cum_profit - curr_max
        perc = dd/curr_max * 100
        return dd, perc

    def max_drawdown(self):
        dd, ddp = self.drawdown()
        return round(dd.min(), 2), round(ddp.min(), 2)
    
    def sharpe_ratio(self):
        n = len(self.df.profit)
        return round(np.sqrt(n) * np.mean(self.df.profit)/np.std(self.df.profit), 2)
    
    def avg_profit(self):
        profit_trades = self.df[self.df.profit > 0]
        return profit_trades.profit.mean()
    
    def avg_loss(self):
        loss_trades = self.df[self.df.profit < 0]
        return loss_trades.profit.mean()
    
    def risk_reward_ratio(self):
        return abs(self.avg_profit()/self.avg_loss())
    
    def long_short_composition(self):
        positive = self.df[self.df.profit > 0].groupby('direction').agg({'profit': 'sum'})
        negative = self.df[self.df.profit < 0].groupby('direction').agg({'profit': 'sum'}).rename(columns={'profit': 'loss'})
        return pd.concat([positive, negative], axis=1).round(2).rename(index={1: 'long', -1: 'short'})
    
    def fund_growth(self, seed=10000):
        """Calculate value of seed amount till end of series.

        Args:
            seed (int, optional): _description_. Defaults to 10000.

        Returns:
            float: non-compounded
            float: compounded
        """
        profit = 0
        curr_cash = seed

        for row in self.df.itertuples():
            profit += int(seed/row.entry_price) * row.profit
            curr_cash += int(curr_cash/row.entry_price) * row.profit

        return round(profit, 2), round(curr_cash, 2)

    def print(self):
        mdd, _ = self.max_drawdown()
        ncom, com = self.fund_growth(seed=self._seed)
        output = f'''
Performance metrics (From: {self.df.iloc[0].entry_time.date()} To: {self.df.iloc[-1].entry_time.date()})

Total profit:           {self.total_profit()}
Peak profit:            {self.peak_profit()}
Win/Loss:               {self.win_vs_loss()[0]}/{self.win_vs_loss()[1]}
Win Rate:               {self.win_rate()} %
Avg. Profit:            {self.avg_profit()}
Avg. Loss:              {self.avg_loss()}
Risk-Reward Ratio:      {self.risk_reward_ratio()}
Max Drawdown:           {mdd}
Sharpe ratio:           {self.sharpe_ratio()}
Fund growth (given {self._seed} seed):
    Simple -            {ncom}
    Compounded -        {com}
'''
        print(output)

    def plot(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Cum. profit", "Daywise profit", "Long Short Split"))
        self.plot_equity_curve(fig)
        self.plot_daywise_profit(fig)
        self.plot_long_short_composition(fig)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            showlegend=False
        )
        fig.show()

    def plot_long_short_composition(self, fig: go.Figure):
        r = self.long_short_composition()
        trace = go.Bar(
            x=r.index,
            y=r.profit,
            text=r.profit,
            textposition="inside",
            textfont_color="white",
            marker_color='rgb(153, 204, 0, 120)'
        )
        fig.add_trace(trace, row=2, col=1)

        trace = go.Bar(
            x=r.index,
            y=r.loss,
            text=r.loss,
            textposition="inside",
            textfont_color="white",
            marker_color='red'
        )
        fig.add_trace(trace, row=2, col=1)

    def plot_daywise_profit(self, fig: go.Figure):
        cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        r = self.df.groupby(self.df.entry_time.dt.day_name()).agg({'profit': 'sum'}).reindex(cats)

        trace = go.Bar(
            x=r.index,
            y=r.profit
        )
        fig.add_trace(trace, row=1, col=2)
        
    def plot_equity_curve(self, fig: go.Figure):
        cum_profit = go.Scatter(
            name="Cumulative Profit",
            x=self.df.entry_time, 
            y=self.df.profit.cumsum(), 
            fill="tozeroy", 
            fillcolor="rgb(153, 204, 0, 120)", 
            line={'color': 'rgb(153, 204, 0, 120)'}, 
        )
        fig.add_trace(cum_profit, row=1, col=1)

        dd, _ = self.drawdown()
        drawdowns = go.Scatter(
            name="Drawdown",
            x=self.df.entry_time,
            y=dd,
            fill="tozeroy",
            fillcolor="rgba(228,128,68,120)",
            line={'color': 'rgba(228,128,68,120)'},
        )
        fig.add_trace(drawdowns, row=1, col=1)

