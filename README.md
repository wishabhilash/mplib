# Installation
Create a virtualenv.
```
pip install mptradelib
```
**Install `pandas_ta` as extra.

# Create strategy
```
mpt create <strategy_name>
```

# Backtest

### Prepare data (example)
```
import pandas as pd
import numpy as np

df = pd.read_csv("something.csv")
df.index = pd.to_datetime(df.datetime)
df.index = df.index.tz_convert('Asia/Kolkata')
df.datetime = df.index
```

Data must have columns
```
datetime, open, high, low, close
```

### Code
```
import pandas_ta as ta
import numpy as np

def cross(ema1, ema2):
    return (ema1 > ema2) & (ema1.shift(1) < ema2.shift(1))

def compute(df, params):
    
    df['ema_fast'] = ta.ema(df.close, params['fast_ema_len'])
    df['ema_slow'] = ta.ema(df.close, params['slow_ema_len'])
    df['ema_trend'] = ta.ema(df.close, params['trend_filter_ema_len'])

    long_cond = (cross(df.ema_fast, df.ema_slow)) & (df.close > df.ema_trend)
    short_cond = (cross(df.ema_slow, df.ema_fast)) & (df.close < df.ema_trend)

    df['long'] = np.where(long_cond, 1, 0)
    df['long_entries'] = np.where((df.long == 1) & (df.long.shift(1) != 1), 1, 0)

    df['short'] = np.where(short_cond, -1, 0)
    df['short_entries'] = np.where((df.short == -1) & (df.short.shift(1) != -1), -1, 0)

    df['entries'] = df.long_entries + df.short_entries
    return df

```

Compute `df['entries']` with `1` for `BUY` and `-1` for `SELL`.

### Run
```
from mptradelib.vectorised_backtest import Backtest

b = Backtest(df, compute)
result = b.run(ema_fast=20, ema_slow=50, ema_trend=200, sl=1, tp=2)
```

`sl` and `tp` are in percentage and mandatory.

params passed in `run` can be accessed using `params` inside `compute`

### Optimize
```
from mptradelib.vectorised_backtest import Backtest

optimization_params = {
    ema_fast: range(1, 20, 1),
    ema_slow: range(20, 50, 1),
    ema_trend: range(100, 200, 1),
    sl: range(0, 1),
    tp: range(1, 10),
}
b = Backtest(df, compute)
result = b.optimize(runs=1, **optimization_params)
```

# Live Trading

### Code
```
import redis
from mptradelib.broker.session import FyersSession
from mptradelib.broker.ticker import LiveTicker
from mptradelib.broker.broker import HistoricalV2
from mptradelib.feed import Datas
from mptradelib.livetrading import BaseStrategy, LiveTrading
import threading
import pandas_ta as ta
import datetime as dt

class MyStrategy(BaseStrategy):
    ema_fast = 20
    ema_slow = 50
    ema_trend = 200
    sl = 1
    tp = 2

    def next(self, symbol, data):
        ema_fast = ta.ema(data.df.close, self.ema_fast)
        ema_slow = ta.ema(data.df.close, self.ema_slow)
        ema_trend = ta.ema(data.df.close, self.ema_trend)

        self.b.buy()

```

### Run
```
mpt runlive <strategy_name> --symbols NSE:SBIN-EQ,NSE:CANB-EQ --param {}
```

### Param
Param can be in two formats-
```
pars = {
        "NSE:SBIN-EQ": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_trend": 200,
            "sl": 1,
            "tp": 2
        }
    }
```
OR
```
{
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_trend": 200,
    "sl": 1,
    "tp": 2
}
```
In later case, same params are used for all symbols.