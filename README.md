# Iterate Torch Metal

Iterative prediction using Pytorch on Apple M1 GPU.

## Iterative maps

A `dynamical system` is a model describing the evolution of an object over time. Using discrete timesteps, the objects temporal evolution is defined by a `rule` or `map` describing the state of the object from time $t$ to time $t+1$. Repeatedly applying a map to an object is called a `iterated map`.

The basic idea of a `iterated map` is to take a number $x_0$ for $t=0$, the `initial condition`, and then in a sequence of $n$ steps to update this number according to a fixed rule or map to obtain a `trajectory`.

## Trajectory predictions

This project explores the prediction of `iterated map` trajectories using Pytorch. The well known `logistic map` is used as the ground truth to train on. Although this `logistic map` is fully deterministic, it's trajectory is chaotic (a-periodic) with certain parameter settings. We'll explore what kind of models in Pytorch suit the modeling of this famous mapping.
