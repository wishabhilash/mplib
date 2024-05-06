class SimulateTrades:
    def __init__(self, trades, initial_cash=100000, slippage=0, tick_size=0.05, leverage=1) -> None:
        self._df = trades
        self._initial_cash = initial_cash
        self._slippage = slippage
        self._tick_size = tick_size
        self._leverage = leverage
    
    def simple(self):
        profit = 0
        for row in self._df.itertuples():
            profit += int((self._initial_cash * self._leverage)/row.entry_price) * (row.profit - self._slippage * self._tick_size * 2)
        return round(profit + self._initial_cash, 2)
    
    def compound(self):
        curr_cash = self._initial_cash
        for row in self._df.itertuples():
            lev_cash = curr_cash * self._leverage
            profit = int(lev_cash/row.entry_price) * (row.profit - self._slippage * self._tick_size * 2)
            curr_cash += profit
        return round(curr_cash, 2)