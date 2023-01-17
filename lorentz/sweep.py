"""
Creates a sweep for hyperparameter tuning using wandb
"""

import argparse
import os
import yaml
import wandb
from types import SimpleNamespace

from train import train

import sys
sys.path.append('..')
import utils.torch_utils as torch_utils


DEFAULT_SWEEP_ARGS = SimpleNamespace(
    sweep_config_path = './sweep_config.yaml',
    count = 10,
    seed = 33,
    dry_run=False
)


def sweep(args=DEFAULT_SWEEP_ARGS):

    # torch_utils.reset_seed(DEFAULT_SWEEP_ARGS.seed)

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    with open(args.sweep_config_path, encoding='utf-8') as file:
        sweep_config = yaml.safe_load(file)
        args.project =  sweep_config['project']

    print((f"using sweep args:\n{yaml.dump(args.__dict__, default_flow_style=False)}").replace('\n', '\n  '))

    # possibly first use CLI wandb login
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, train, count=args.count)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--sweep_config_path', type=str, default=DEFAULT_SWEEP_ARGS.sweep_config_path, help="sweep config path")
    parser.add_argument('-c', '--count', type=int, default=DEFAULT_SWEEP_ARGS.count, help='the number (count) of sweep runs')
    parser.add_argument('-q', '--dry_run', default=DEFAULT_SWEEP_ARGS.dry_run, action='store_true', help='Dry run (do not log to wandb)')  

    args = parser.parse_args()

    sweep(args)
