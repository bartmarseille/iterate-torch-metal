'''
 perform the training with hyperparameters as its arguments
 '''

import argparse # build in Python module for just the thing!
import os
import yaml

from types import SimpleNamespace

from train import train



# default location of configuration file
DEFAULT_CONFIG_PATH='./default_config.yaml'


# parse all args and call the model
if __name__ == "__main__":

    # exclude non-passed arguments
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # required config values (have default value)
    parser.add_argument('-p', '--config_path', type=str, default=DEFAULT_CONFIG_PATH, help="config path")
    parser.add_argument('-q', '--dry_run', default=False, action='store_true', help='Dry run (do not log to wandb)')  

    # optional config values (if present, they overrule the config_path file values)
    parser.add_argument('-pr', '--project', type=str, help="main project name")
    parser.add_argument('-s', '--seed', type=int, help="random seed for reproducibility")
    parser.add_argument('-hu', '--hidden_units_list', nargs='+', type=int, help='list of hidden units per layer')
    parser.add_argument('-a', '--activation_function', type=str, help='activation function')
    parser.add_argument('-b', '--batch_size', type=int, help="batch size")
    parser.add_argument('-e', '--epochs', type=int, help='number of training epochs')
    parser.add_argument('-n', '--dataset_size', type=int, help='number of training examples in train dataset')
    parser.add_argument('-lr', '--learning_rate', type=float,  help="learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float,  help="weight decay")

    args = parser.parse_args()

    print((f"using config args:\n{yaml.dump(args.__dict__, default_flow_style=False)}").replace('\n', '\n  '))

     # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    # load config
    with open(args.config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # use config but allow args to overrule values
    config = {**config, **vars(args)}
    
    train(config=config)

    