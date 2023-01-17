"""
Defines a simple NN model on the lorentz ODE based dataset.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from test import test

from tqdm import tqdm
import wandb
import yaml
from types import SimpleNamespace

import sys
sys.path.append('..')

import utils.ode_utils as ode_utils
import utils.torch_utils as torch_utils


class RecurrenceModel(nn.Module):

    def name(self):
        return self._get_name()

    def __init__(self, hidden_units_list: list, activation_function='relu'):
        super(RecurrenceModel, self).__init__()
        
        if hidden_units_list[0]!=hidden_units_list[-1]:
            raise ValueError(f'number of input ({hidden_units_list[0]}) and output features ({hidden_units_list[-1]}) must be the same for Recurrence model {hidden_unit_list}')

        self.num_layers = len(hidden_units_list)-1
        self.activation_func = getattr(F, activation_function)

        # print(f'using {activation_function} activation function')

        for layer_num, in_features, out_features in zip(range(self.num_layers), hidden_units_list, hidden_units_list[1:]):
            layer = nn.Linear(in_features, out_features, bias=True)
            setattr(self, f'fc{layer_num}', layer)

    def forward(self, x):
        for layer_num in range (self.num_layers):
            layer = getattr(self, f'fc{layer_num}')
            
            if layer_num < self.num_layers -1:
                x = self.activation_func(layer(x))
            else:
                # last layer has no activation function
                x = layer(x)
        return x 


def create_dataset(num_samples: int, device: str, **kwargs):
    '''
    prepare the data
    '''
    
    mu, sigma = kwargs['x']['mu'], kwargs['x']['sigma'] # mean and standard deviation
    x = np.random.normal(mu, sigma, num_samples)

    mu, sigma = kwargs['y']['mu'], kwargs['y']['sigma']
    y = np.random.normal(mu, sigma, num_samples)

    mu, sigma = kwargs['z']['mu'], kwargs['z']['sigma']
    z = np.random.normal(mu, sigma, num_samples)

    X = np.vstack((x, y, z)).T.reshape(-1,3)
    # shuffle
    # np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)

    # get the target values Y
    Y = ode_utils.rk4(ode_utils.lorentz_ode, X, t=1.0, **kwargs)

    # put X and Y in dataset, suited for the device (Mps)
    dataset = torch_utils.MpsDataset(X, Y, device)
    
    return dataset



def train(config: dict=None, config_path: str='./default_config.yaml'):

    if not config:
        with open(config_path, encoding='utf-8') as file:
            config = yaml.safe_load(file)

    

    # start a run with config
    run=wandb.init(project=config['project'])
    # set config as default, will be overrulled  by sweep config
    run.config.setdefaults(config)
    # get final config, possibly altered by sweep
    config = SimpleNamespace(**run.config)
    
    # reset seed for reproducibility
    torch_utils.reset_seed(config.seed)

    # create random distributed train and dev dataset
    train_dataset = create_dataset(config.dataset_size, config.device, **config.dataset_parameters)
    dev_dataset = create_dataset(1000, config.device, **config.dataset_parameters)
    X_dev, Y_dev = dev_dataset.get_data()
    # create dataloader for train dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # init model
    model = RecurrenceModel(config.hidden_units_list, config.activation_function)
    model.to(config.device)

    # Define loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    if hasattr(config, 'weight_decay'):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    num_train_examples = 0
    for epoch in range(config.epochs):
        with tqdm(train_loader, unit=" batch") as tepoch:
            for step, (X, Y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{config.epochs}")
                
                X, Y = X.to(device=config.device), Y.to(device=config.device)
                optimizer.zero_grad()
                Y_hat = model(X)
                loss = torch.sqrt(loss_fn(Y_hat, Y))
                loss.backward()
                optimizer.step()

                num_train_examples += len(Y)
                tepoch.set_postfix(loss=loss.item(), examples=f'{round(num_train_examples/1000, 0)}K')

                if step % 100 == 0: 
                    # calculate dev-set loss
                    Y_dev_hat = model(X_dev)
                    dev_loss = torch.sqrt(loss_fn(Y_dev_hat, Y_dev))

                    wandb.log({'train_loss': loss.item(), 'dev_loss': dev_loss.item()})
                                
    test_loss = test(model, config.map_parameters, config.device)
    wandb.log({'test_loss': test_loss, 'total_train_examples': num_train_examples})

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    #finish run
    run.finish()
