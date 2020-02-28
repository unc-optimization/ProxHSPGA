# ProxHSPGA

## Introduction
This package is the implementation of ProxHSPGA and its non-composite variant to solve several optimization problems in reinforcement learning.

## Dependency

1. Install `rllab` following instructions from [rllab installation](https://rllab.readthedocs.io/en/latest/user/installation.html).

2. Install latest version of `Theano` and `Lasagne`, somehow the stable versions do not work.
```
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

**Note**

For Mujoco environments, you need to acquire a license for mujoco-py installation.

## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software you are highly encouraged to cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, P. H. Nguyen, M. van Dijk, and Q. Tran-Dinh, **A Hybrid Stochastic Policy Gradient Algorithm for Reinforcement Learning**, The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), Palermo, Italy, 2020.

Feel free to send feedback and questions about the package to our maintainer Nhan H. Pham at <nhanph@live.unc.edu>.