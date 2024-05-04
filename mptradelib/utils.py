import datetime
import threading
import pandas as pd

def iterdaterange(step=10):
    now = datetime.datetime.now()
    while True:
        end_date = now.strftime("%Y-%m-%d")
        then = now - datetime.timedelta(days=step)
        start_date = then.strftime("%Y-%m-%d")
        yield start_date, end_date
        now = then - datetime.timedelta(days=1)

def threaded(func):
    """
    Decorator that multithreads the target function
    with the given parameters. Returns the thread
    created for the function
    """
    def wrapper(*args):
        thread = threading.Thread(target=func, args=args)
        thread.start()
        return thread
    return wrapper

class Tearsheet:
    def __init__(self, result: pd.DataFrame) -> None:
        self.df = result

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
    
    def drawdown_perc(self):
        prices = self.df.profit
        cumulative_returns = (prices / prices.iloc[0]) - 1
        peak = cumulative_returns.cummax()
        return (cumulative_returns - peak) / peak
    
    def drawdown(self):
        dp = self.drawdown_perc()
        return dp * self.df.profit.iloc[0]

    def max_drawdown(self):
        return round(self.drawdown().min(), 2)
    
    def max_drawdown_perc(self):
        return round(self.drawdown_perc().min(), 2)

    def sharpe_ratio(self):
        n = len(self.df.profit)
        return round(np.sqrt(n) * np.mean(self.df.profit)/np.std(self.df.profit), 2)
    

    def print(self):
        output = f'''
Performance metrics
Total profit:    {self.total_profit()}
Peak profit:     {self.peak_profit()}
Win/Loss:        {self.win_vs_loss()[0]}/{self.win_vs_loss()[1]}
Win Rate:        {self.win_rate()} %
Max Drawdown:    {self.max_drawdown()} or {self.max_drawdown_perc()}%
Sharpe ratio:    {self.sharpe_ratio()}
'''
        print(output)

    def plot(self):
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            print("please install plotly by 'pip install plotly'")
            return
        
        fig = make_subplots(rows=1, cols=1)
        cum_profit = go.Scatter(x=self.df.entry_time, y=self.df.profit.cumsum())
        fig.add_trace(cum_profit)

        drawdowns = go.Scatter(x=self.df.entry_time, y=self.drawdown())
        fig.add_trace(drawdowns)
        fig.show()

