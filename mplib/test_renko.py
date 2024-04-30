import pandas as pd
import pytest
from datetime import datetime
from . import renko
import numpy as np


# Helper function to create a sample price series
def create_price_series(start_date, end_date, frequency, brick_size):
    date_rng = pd.date_range(start=start_date, end=end_date, freq=frequency)
    price_data = [100, 105, 98, 110, 115, 92, 105, 98, 110, 115]
    price_series = pd.Series(price_data, index=date_rng)
    return price_series


# Test cases
def test_generate_renko_bricks_valid_input():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 10)
    frequency = "D"
    brick_size = 5

    price_series = create_price_series(start_date, end_date, frequency, brick_size)

    r = renko.Renko(price_series, 5)
    r.update(price_series, 5)

    # price_data = [100, 105, 98, 110, 115, 92, 105, 98, 110, 115]
    expected = pd.DataFrame(
        {
            "Open": [100, 105, 110, 110, 105, 100, 100, 105, 110],
            "Close": [105, 110, 115, 105, 100, 95, 105, 110, 115],
            "High": [105, 110, 115, 110, 105, 100, 100, 110, 115],
            "Low": [100, 105, 110, 105, 100, 95, 100, 105, 110],
        }
    )

    assert (r.renko_bricks.Open.to_list() == expected.Open.to_list())
    assert (r.renko_bricks.Close.to_list() == expected.Close.to_list())
    
