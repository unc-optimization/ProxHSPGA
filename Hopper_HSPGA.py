"""@package Hopper_HSPGA

This package implements the HSPGA algorithm for Hopper-v2 environment.

Copyright (c) 2020 Nhan H. Pham, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2020 Lam M. Nguyen, IBM Research, Thomas J. Watson Research Center
Yorktown Heights

Copyright (c) 2020 Dzung T. Phan, IBM Research, Thomas J. Watson Research Center
Yorktown Heights

Copyright (c) 2020 Phuong Ha Nguyen, Department of Electrical and Computer Engineering, University of Connecticut

Copyright (c) 2020 Marten van Dijk, Department of Electrical and Computer Engineering, University of Connecticut

Copyright (c) 2020 Quoc Tran-Dinh, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill
All rights reserved.

If you found this helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, P. H. Nguyen, M. van Dijk and Q. Tran-Dinh, **A Hybrid Stochastic Policy Gradient Algorithm for Reinforcement Learning**, The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), Palermo, Italy, 2020.

"""

from rllab.envs.gym_env import GymEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import adam
import pandas as pd
import os

from utils.utils import *
    
# whether to load existing policy
load_policy=True

# whether to learn standard deviation of Gaussian policy
learn_std = False

# snapshot batchsize
snap_bs = 50

# effective length of a trajectory
traj_length = 500

# minibatch size in the inner loop
m_bs = 5

# number of trajectories for evaluation
num_eval_traj = 50

# discount factor
discount = 0.99

# stepsizes
learning_rate = 5e-4
beta = 0.99

# number of inner iterations
max_inner = 3

# total number of trajectories
max_num_traj = 10000

# normalize() makes sure that the actions for the environment lies
# within the range [-1, 1] (only works for environments with continuous actions)
env = normalize(GymEnv("Hopper-v2"))

# initialize a neural network policy with a single hidden layer of 8 hidden units
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32), learn_std=learn_std)
prev_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,32), learn_std=learn_std)

# policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
# distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
# the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
# rllab.distributions.DiagonalGaussian
dist = policy.distribution
prev_dist = prev_policy.distribution

# create placeholders
observations_var = env.observation_space.new_tensor_variable(
    'observations',
    # It should have 1 extra dimension since we want to represent a list of observations
    extra_dims=1
)

actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)

d_rewards_var = TT.vector('d_rewards')

importance_weights_var = TT.vector('importance_weight')

# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
dist_info_vars = policy.dist_info_sym(observations_var)
prev_dist_info_vars = prev_policy.dist_info_sym(observations_var)

params = policy.get_params(trainable=True)
prev_params = prev_policy.get_params(trainable=True)

importance_weights = dist.likelihood_ratio_sym(actions_var,dist_info_vars,prev_dist_info_vars)

# create surrogate losses
surr_on1 = TT.mean(- dist.log_likelihood_sym(actions_var,dist_info_vars)*d_rewards_var)
surr_on2 = TT.mean(prev_dist.log_likelihood_sym(actions_var,prev_dist_info_vars)*d_rewards_var*importance_weights_var)
grad = theano.grad(surr_on1, params)
grad_diff = [sum(x) for x in zip(theano.grad(surr_on1,params),theano.grad(surr_on2,prev_params))]

print("Parameters shapes")
for i in range(len(params)):
    print(params[i].shape.eval())

eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
eval_grad3 = TT.matrix('eval_grad2',dtype=grad[2].dtype)
eval_grad4 = TT.vector('eval_grad3',dtype=grad[3].dtype)
eval_grad5 = TT.matrix('eval_grad4',dtype=grad[4].dtype)
eval_grad6 = TT.vector('eval_grad5',dtype=grad[5].dtype)
if learn_std:
    eval_grad7 = TT.vector('eval_grad5',dtype=grad[6].dtype)

f_compute_grad = theano.function(
    inputs = [observations_var, actions_var, d_rewards_var],
    outputs = grad
)

if learn_std:
    f_update = theano.function(
        inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7],
        outputs = None,
        updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7], params, learning_rate=learning_rate)
    )
else:
    f_update = theano.function(
    inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6],
    outputs = None,
    updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6], params, learning_rate=learning_rate)
)

f_importance_weights = theano.function(
    inputs = [observations_var, actions_var],
    outputs = importance_weights
)

f_compute_grad_diff = theano.function(
    inputs=[observations_var, actions_var, d_rewards_var, importance_weights_var],
    outputs=grad_diff,
)

# log directory
log_dir = "log_file/Hopper/HSPGA"+"_lr" + str(learning_rate)

# check if directory exists, if not, create directory
if not os.path.exists( log_dir ):
    os.makedirs( log_dir )

# setup parallel sampler
parallel_sampler.populate_task(env, policy)
parallel_sampler.initialize(8)

# initialize log Data Frame
avg_return_data = pd.DataFrame()

for k in range(10):

    print("Run #{}".format(k))

    # load policy
    if learn_std:
        file_name = 'hopper_policy' + '.txt'
    else:
        file_name = 'hopper_policy_novar' + '.txt'

    if load_policy:
        policy.set_param_values(np.loadtxt('save_model/' + file_name), trainable=True)       
    else:
        np.savetxt("save_model/" + file_name,policy.get_param_values(trainable=True))
        load_policy = True

    # intial setup
    avg_return = list()
    eps_list = []
    max_rewards = -np.inf
    num_traj = 0

    # loop till done
    while num_traj <= max_num_traj:
        # sample snapshot batch of trajectories
        paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),snap_bs,traj_length,show_bar=False)
        paths = paths[:snap_bs]

        # extract information
        observations, actions, d_rewards = extract_path(paths, discount)

        # compute policy gradient
        v_est = compute_snapshot_grad_est(f_compute_grad, observations, actions, d_rewards)

        # perform update
        if learn_std:
            f_update(v_est[0],v_est[1],v_est[2],v_est[3],v_est[4],v_est[5],v_est[6])
        else:
            f_update(v_est[0],v_est[1],v_est[2],v_est[3],v_est[4],v_est[5])

        # sample trajectories for evaluating current policy
        tmp_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),num_eval_traj,show_bar=False)

        avg_return.append(np.mean([sum(p["rewards"]) for p in tmp_paths]))
        eps_list.append(num_traj)
        print(str(num_traj)+' Average Return:', avg_return[-1])

        # update best policy
        if avg_return[-1] > max_rewards:
            max_rewards = avg_return[-1]
            best_policy_ = policy.get_param_values(trainable=True)

        # update number of trajectories sampled
        num_traj += snap_bs

        # inner loop
        for _ in range(max_inner):
            # sample trajectories
            sub_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),2*m_bs,traj_length,show_bar=False)
            sub_paths = sub_paths[:2*m_bs]

            # update number of trajectories sampled
            num_traj += 2*m_bs

            sub_paths_1 = sub_paths[0:m_bs-1]
            sub_paths_2 = sub_paths[m_bs:2*m_bs-1]

            # extract information
            sub_observations_1, sub_actions_1, sub_d_rewards_1 = extract_path(sub_paths_1, discount)
            sub_observations_2, sub_actions_2, sub_d_rewards_2 = extract_path(sub_paths_2, discount)

            path_info_1 = {
                'obs': sub_observations_1,
                'acts': sub_actions_1,
                'rws': sub_d_rewards_1,
            }

            path_info_2 = {
                'obs': sub_observations_2,
                'acts': sub_actions_2,
                'rws': sub_d_rewards_2,
            }

            # compute Hybrid SPG estimator
            v_est = compute_hybrid_spg_est(f_compute_grad,f_compute_grad_diff,f_importance_weights,path_info_1,path_info_2,beta,v_est)

            # perform update
            prev_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True) 
            if learn_std:
                f_update(v_est[0],v_est[1],v_est[2],v_est[3],v_est[4],v_est[5],v_est[6])
            else:
                f_update(v_est[0],v_est[1],v_est[2],v_est[3],v_est[4],v_est[5])

            # check if we are done
            if num_traj >= max_num_traj:
                tmp_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),num_eval_traj,show_bar=False)

                avg_return.append(np.mean([sum(p["rewards"]) for p in tmp_paths]))
                eps_list.append(num_traj)
                print(str(num_traj)+' Average Return:', avg_return[-1])

                # update best policy
                if avg_return[-1] > max_rewards:
                    max_rewards = avg_return[-1]
                    best_policy_ = policy.get_param_values(trainable=True)
                    
                break   

    # log data
    if k==0:
        avg_return_data["Episodes"]=eps_list
    avg_return_data["MeanRewards_"+str(k)]=avg_return

    avg_return_df = pd.DataFrame()
    avg_return_df["Episodes"]=eps_list
    avg_return_df["MeanRewards"]=avg_return
    avg_return_df.to_csv(os.path.join(log_dir,"avg_return_" + str(k) + ".csv"), index=False)

    np.savetxt(os.path.join(log_dir,"final_policy_"+str(k) + ".txt"),policy.get_param_values(trainable=True))
    np.savetxt(os.path.join(log_dir,"best_policy_"+str(k) + ".txt"),best_policy_)

print(avg_return_data)
avg_return_data.to_csv(os.path.join(log_dir,"avg_return_total.csv"),index=False)

