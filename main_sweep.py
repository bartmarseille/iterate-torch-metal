from __future__ import print_function

import argparse

import math
import numpy as np
import time

from types import SimpleNamespace
from tqdm import tqdm

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import yaml

import utils.torch_utils as torch_utils
import utils.ode_utils as ode_utils



class RecurrenceModel(nn.Module):

    def name(self):
        return self._get_name()

    def __init__(self, hidden_units_list: list, activation_function='relu'):
        super(RecurrenceModel, self).__init__()
        
        # TODO get from config: 
        # config.hidden_units_list
        # config.activation_function
        if hidden_units_list[0]!=hidden_units_list[-1]:
            raise ValueError(f'number of input ({hidden_units_list[0]}) and output features ({hidden_units_list[-1]}) must be the same for Recurrence model {hidden_unit_list}')

        self.num_layers = len(hidden_units_list)-1
        self.activation_func = getattr(F, activation_function)

        print(f'using {activation_function} activation function')

        for layer_num, in_features, out_features in zip(range(self.num_layers), hidden_units_list, hidden_units_list[1:]):
            layer = nn.Linear(in_features, out_features, bias=True)
            setattr(self, f'fc{layer_num}', layer)

    def forward(self, x):
        for layer_num in range (self.num_layers):
            layer = getattr(self, f'fc{layer_num}')
            
            if layer_num < self.num_layers -1:
                x = self.activation_func(layer(x))
            else:
                #last layer has no activation function    
                x = layer(x)
        return x


# def train_dataloader(config):
#     torch_utils.reset_seed(config.seed)

#     # create random distributed train dataset
#     mu, sigma = 0, 30 # mean and standard deviation
#     x = np.random.normal(mu, sigma, config.num_samples)
#     y = np.random.normal(mu, sigma, config.num_samples)
#     mu, sigma = 25, 25 # mean and standard deviation
#     z = np.random.normal(mu, sigma, config.num_samples)

#     X_train = np.vstack((x,y,z)).T.reshape(-1,3)
#     # shuffle
#     # np.take(X_train, np.random.rand(X_train.shape[0]).argsort(), axis=0, out=X_train)

#     # get the target values Y
#     Y_train = ode_utils.rk4(ode_utils.lorentz_ode, X_train, t=1.0, **config.parameters)

#     # put X and Y in dataset, suited for Apple M1 GPU (Mps)
#     train_dataset = torch_utils.MpsDataset(X_train, Y_train, config.device)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

#     return train_loader


def train(config):

    default_config = SimpleNamespace(
        # reproducability
        seed=33,
        
        # train model
        model_name="RecurrenceModel",
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
        learning_rate=0.02305,  # hyper par tuning
        weight_decay=0.006251,  # hyper par tuning
        loss_fuc="RMSE",

        # device
        device="mps",
        dtype="float32",
        # num_workers=4,
        gpu_name="M1Pro GPU 16 Cores",
    )

    run = wandb.init(config=default_config)
    config = wandb.config


    start = time.time()
    torch_utils.reset_seed(config.seed)

    # create random distributed train and dev dataset
    mu, sigma = 0, 30 # mean and standard deviation
    x_train = np.random.normal(mu, sigma, config.num_samples)
    y_train = np.random.normal(mu, sigma, config.num_samples)
    dev_num_samples = 1000
    x_dev = np.random.normal(mu, sigma, dev_num_samples)
    y_dev = np.random.normal(mu, sigma, dev_num_samples)
    mu, sigma = 25, 25 # mean and standard deviation
    z_train = np.random.normal(mu, sigma, config.num_samples)
    z_dev  = np.random.normal(mu, sigma, dev_num_samples)

    X_train = np.vstack((x_train,y_train,z_train)).T.reshape(-1,3)
    X_dev = np.vstack((x_dev,y_dev,z_dev)).T.reshape(-1,3)
    # shuffle
    # np.take(X_train, np.random.rand(X_train.shape[0]).argsort(), axis=0, out=X_train)

    # get the target values Y
    Y_train = ode_utils.rk4(ode_utils.lorentz_ode, X_train, t=1.0, **config.parameters)
    Y_dev = ode_utils.rk4(ode_utils.lorentz_ode, X_dev, t=1.0, **config.parameters)

    # put X and Y in dataset, suited for Apple M1 GPU (Mps)
    train_dataset = torch_utils.MpsDataset(X_train, Y_train, config.device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataset = torch_utils.MpsDataset(X_dev, Y_dev, config.device)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=dev_num_samples, shuffle=False)

    # init model
    model = RecurrenceModel(config.hidden_units_list, config.activation_function)
    model.to(config.device)

    # Make the loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')

    if hasattr(config, 'weight_decay'):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    train_loss=[]
    dev_loss = []
    test_loss=[]

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


                    wandb.log({"loss": loss})
                train_loss.append(loss.item())
        
        _, _, test_loss = test(model, config)
        wandb.log({"test_loss": test_loss})

    # calculate timesteps for the ODE trajectory
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)
    epoch_time = [x/n_steps_per_epoch for x in range(len(train_loss))]

    end = time.time()
    wandb.log({"total_train_examples": num_train_examples})
    print(f'training time: {end - start: .2f} sec')

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

    # return model, epoch_time, train_loss


def test(model, config):
    P = config.parameters
    x0 = [1.5, 0.6, 0.7]

    X = ode_utils.iterate(ode_utils.lorentz_ode, x0, n=3501, **P)
    Y_dot = X[1:,:]
    X = X[:-1,:]

    X_torch = torch.tensor(X, dtype=torch.float32).to(config.device)
    Y_hat = model(X_torch)
    Y_hat = Y_hat.cpu().detach().numpy()

    rmse_loss = np.sqrt(np.mean((Y_hat-Y_dot)**2))

    wandb.log({"test_loss": rmse_loss})

    return Y_dot, Y_hat, rmse_loss
    


def main():

    # Set up your default hyperparameters
    with open('./sweep_config.yaml') as file:
        # sweep_config = yaml.load(file, loader=yaml.FullLoader)
        try:
            sweep_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    sweep_id = wandb.sweep(sweep_config, project="leaf-cnn2")

    wandb.agent(sweep_id, train, count=20)


    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # wandb.config.update(args)

    # torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       momentum=args.momentum)
    # wandb.watch(model)

    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(args, model, device, test_loader)


if __name__ == '__main__':
    # possibly first use CLI wandb login
    main()