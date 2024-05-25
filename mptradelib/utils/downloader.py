import datetime as dt
from tqdm import tqdm
import os
import pandas as pd
from mptradelib.broker.broker import Historical
from functools import partial
from retry import retry


@retry(tries=10, delay=2)
def worker(h, scrip, start, end, resolution, download_dir):
    filename = f'{download_dir}/{scrip}-{resolution}.parquet'
    if os.path.exists(filename):
        dfe = pd.read_parquet(filename)
        if not dfe.empty:
            start = dfe.tail(1).iloc[0].datetime + dt.timedelta(days=1)
            if start.date() > end.date():
                return
        else:
            os.remove(filename)
            print('removed ', filename)
    try:
        df = h.historical(scrip, resolution=resolution, start=start, end=end)
    except KeyError as e:
        return
    if os.path.exists(filename) and not (dfe.empty or df.empty):
        df = pd.concat([df, dfe])
    df.to_parquet(filename)

def download_historical(h: Historical, scrips: list[str], start, end, resolution=5, download_dir="data"):
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    with tqdm(total=len(scrips)) as pb:
        for scrip in scrips:
            f = partial(worker, h, start=start, end=end, resolution=resolution, download_dir=download_dir)
            f(scrip)
            pb.update()
            