import pandas as pd

class SimulateTrades:
    profits = []
    
    def __init__(self, trades, initial_cash=100000, slippage=0, tick_size=0.05, leverage=1) -> None:
        self._df = trades
        self._initial_cash = initial_cash
        self._slippage = slippage
        self._tick_size = tick_size
        self._leverage = leverage
    
    def run(self):
        opening_balance = self._initial_cash
        tradable_fund = opening_balance
        last_date = None
        margin = 0
        for row in self._df.itertuples():
            if row.entry_time.date() != last_date:
                if last_date is not None:
                    self.profits.append(tradable_fund + margin - opening_balance)
                last_date = row.entry_time.date()
                opening_balance = tradable_fund + margin
            total_profit = round(int(self._leverage * tradable_fund/row.entry_price) * (row.profit - self._slippage * self._tick_size * 2), 2)
            if total_profit >= 0:
                tradable_fund += total_profit
                if tradable_fund > opening_balance:
                    margin = tradable_fund - opening_balance
                    tradable_fund = opening_balance
            else:
                margin += total_profit
                if margin < 0:
                    tradable_fund += margin
                    margin = 0
        self.profits.append(tradable_fund + margin - opening_balance)
        return pd.concat([self._df, pd.DataFrame({'sim_profit': self.profits})], axis=1)
