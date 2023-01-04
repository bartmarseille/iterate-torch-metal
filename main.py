'''
 perform the training with hyperparameters as its arguments
 '''

import argparse # build in Python module for just the thing!
import os
import yaml

from types import SimpleNamespace

from train import train

# def get_default_hyperparameters(path='./sweep_config.yaml'):
#     default_hyperpars = None
#     # Set up your default hyperparameters
#     with open(path=path) as file:
#         try:
#             default_hyperpars = yaml.safe_load(file)
#         except yaml.YAMLError as e:
#             print(e)
#     return default_hyperpars


# define the default hyperparameter values
CONFIG = SimpleNamespace(

    project='CLI-test',

    # reproducability
    seed=33,
    
    # train model
    model_name='RecurrenceModel',
    hidden_units_list = [3,8,3],
    activation_function = 'leaky_relu',
    
    # train regime
    batch_size=32,  # 32 better than 64 better than 128
    epochs=5,

    # train data
    dataset="np.random.normal([x:0,30, y:0,30, z:25,25], num_samples)",
    map_name="lorentz_ode",
    parameters = {'sigma': 10, 'beta': 8/3, 'rho': 28, 'dt': 0.01},
    num_samples = 20**3,  # 30**3 better than 40**3
    
    optimizer="Adam",
    learning_rate=0.02305,  # hyperpar tuning
    weight_decay=0.006251,  # hyperpar tuning
    loss_fuc="RMSE",

    # device
    device="mps",
    dtype="float32",
    # num_workers=4,
    gpu_name="M1Pro GPU 16 Cores",
)


# parse all args and call the model
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--project_name', type=str, default=CONFIG.project, help="main project name")

    parser.add_argument('--seed', type=str, default=CONFIG.seed, help="random seed")

    parser.add_argument('--hidden_units_list', type=list[int], default=CONFIG.hidden_units_list, help='list of hidden units per layer')

    parser.add_argument('--activation_function', type=str, default=CONFIG.activation_function, help='activation function')

    parser.add_argument('-b', '--batch_size', type=int, default=CONFIG.batch_size, help="batch size")

    parser.add_argument('-e', '--epochs', type=int, default=CONFIG.epochs, help='number of training epochs')

    parser.add_argument('-n', '--num_samples', type=int, default=CONFIG.num_samples, help='number of training examples')

    parser.add_argument('-lr', '--learning_rate', type=float, default=CONFIG.learning_rate,  help="learning rate")

    parser.add_argument('--weight_decay', type=float, default=CONFIG.weight_decay,  help="weight decay")

    parser.add_argument('-q', '--dry_run', action='store_true', help='Dry run (do not log to wandb)')  


    # use parse_args or otherwise DEFAULT_CONFIG 
    args = SimpleNamespace(**{**vars(CONFIG), **vars(parser.parse_args())})

    # easier testing--don't log to wandb if dry run is set
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    train(args)

    