from pytest import mark
import os
from pathlib import Path
from pprint import pprint
import argparse
from catalyst.utils.misc import boolean_flag
from catalyst.dl.scripts.run import parse_args_uargs
from catalyst.utils.sys import get_config_runner
import shutil

def test_classification_config_runner(config_list):
    for i in config_list:
        try:
            shutil.rmtree('./logs/')
        except FileNotFoundError:
            pass
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            "--configs",
            "-C",
            nargs="+",
            help="path to config/configs",
            metavar="CONFIG_PATH",
            dest="configs",
            required=False,
        )
        boolean_flag(parser, "check", default=None)

        args, unknown_args = parser.parse_known_args(['--config', f'src/metric_learning/tests/test_configs/metric_learning/{i}', '--check'])

        args, config = parse_args_uargs(args, unknown_args)
        runner: ConfigRunner = get_config_runner(expdir=args.expdir, config=config)
        runner.run()

@mark.test_metric_learning
def test_metric_learning_configs():
    path = Path('src/metric_learning/tests/test_configs').absolute() / 'metric_learning/'
    config_list = os.listdir(str(path))
    test_classification_config_runner(config_list)

