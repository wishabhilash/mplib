import datetime
import threading


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