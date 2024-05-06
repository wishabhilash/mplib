import datetime
import threading
import pandas as pd
import numpy as np
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("please install plotly by 'pip install plotly' to enable plotting")

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

