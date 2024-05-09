import click
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
import sys
import logging
import shutil
from importlib import import_module

logger = logging.getLogger(__name__)
logging.addLevelName(logging.INFO, 'info')

libpath = os.path.abspath(os.path.join(sys.executable, f"../../lib/python{'.'.join(sys.version.split('.')[:2])}/site-packages/mptradelib"))
if not os.path.exists(libpath):
    libpath = os.path.join(os.path.abspath('.'), "mptradelib")

env = Environment(
    loader=FileSystemLoader(os.path.join(libpath, "cli/templates/strategy/")),
    extensions=['jinja2_strcase.StrcaseExtension']
)

@click.group
def commands():
    pass

@click.command()
@click.argument("strategy-name")
def create(strategy_name):
    if not os.path.exists(strategy_name):
        os.mkdir(strategy_name)

    for f in env.list_templates():
        try:
            template = env.get_template(f)
        except Exception as e:
            logger.error("template not found")
            shutil.rmtree(strategy_name)
            return

        rendered_file_name = '.'.join(f.split('.')[:-1])
        rendered_file_path = os.path.join(strategy_name, rendered_file_name)
        if os.path.exists(rendered_file_path):
            ans = click.prompt(f"do u want to overwrite {rendered_file_name}? (yes/no)", default='no')
            if ans == 'no':
                continue

        with open(rendered_file_path, 'w') as f:
            f.write(template.render(strategy_name=strategy_name))
    print("strategy created!!!")

@click.command()
@click.argument("strategy-name")
@click.option('--symbols', type=str, default="", help="Comma separated stock symbols")
@click.option('--timeframe', default=1, help="Comma separated stock symbols")
@click.option('--params', default={}, help="Strategy parameters")
def runlive(strategy_name, symbols, timeframe, params):
    if not symbols:
        raise ValueError("no symbols found")
    try:
        mod = import_module(f'{strategy_name}')
        symbols = symbols.split(',')
        mod.live(symbols, timeframe, params)
    except Exception as e:
        if str(e) == "No module named 'pandas'":
            print("do 'pip install pandas' for live trading to work.")
        else:
            print(e)
        return

commands.add_command(create)
commands.add_command(runlive)