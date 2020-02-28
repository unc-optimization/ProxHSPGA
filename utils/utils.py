"""@package utils

This package implements useful functions.

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

import numpy as np

def extract_path(paths, discount):
    for path in paths:
        p_rewards = path["rewards"]

        returns = []
        return_so_far = 0
        for t in range(len(p_rewards) - 1, -1, -1):
            return_so_far = p_rewards[t] + discount * return_so_far
            returns.append(return_so_far)

        # reverse return array
        returns = np.array(returns[::-1])

        # add to dict
        path["returns"] = returns

    # d_rewards_tmp = [p["rewards"] for p in paths]
    
    # d_rewards_tmp = calc_discount_rewards(d_rewards_tmp, discount)

    observations = [p["observations"] for p in paths]
    actions = [p["actions"] for p in paths]
    d_rewards = [p["returns"] for p in paths]

    # print(d_rewards[0],'\n',d_rewards_tmp[0])

    return observations, actions, d_rewards

def prox_l1_norm( w, lamb ):
    """! Compute the proximal operator of the \f$\ell_1\f$ - norm

    \f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
    
    Parameters
    ---------- 
    @param w : input vector
    @param lamb : penalty paramemeter
        
    Returns
    ---------- 
    @retval : perform soft - thresholding on input vector
    """
    return np.sign( w ) * np.maximum( np.abs( w ) - lamb, 0 )

def prox_l2_square(x, lbd):
    """! Compute the proximal operator of the \f$\ell_2\f$ - norm

    \f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_2^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
    
    Parameters
    ---------- 
    @param w : input vector
    @param lamb : penalty paramemeter
        
    Returns
    ---------- 
    @retval : perform soft - thresholding on input vector
    """
    return (1.0 / (1.0 + lbd)) * x

def compute_snapshot_grad_est(f_compute_grad, obs, acts, rws):
    # compute policy gradient
    v_est = f_compute_grad(obs[0], acts[0], rws[0])
    for ob,ac,rw in zip(obs[1:],acts[1:],rws[1:]):
        g_i = f_compute_grad(ob, ac, rw)
        v_est = [sum(x) for x in zip(v_est,g_i)]
    v_est = [x/len(obs) for x in v_est]

    return v_est

def compute_hybrid_spg_est(f_compute_grad,f_compute_grad_diff,f_importance_weights,path_info_1,path_info_2,beta,v_est):
    sub_observations_1  = path_info_1['obs']
    sub_actions_1       = path_info_1['acts']
    sub_d_rewards_1     = path_info_1['rws']

    sub_observations_2  = path_info_2['obs']
    sub_actions_2       = path_info_2['acts']
    sub_d_rewards_2     = path_info_2['rws']

    iw = f_importance_weights(sub_observations_1[0],sub_actions_1[0])
    grad_diff = f_compute_grad_diff(sub_observations_1[0],sub_actions_1[0],sub_d_rewards_1[0],iw)
    u_est = f_compute_grad(sub_observations_2[0],sub_actions_2[0],sub_d_rewards_2[0])

    for ob_1,ac_1,rw_1,ob_2,ac_2,rw_2 in zip(sub_observations_1[1:],sub_actions_1[1:],sub_d_rewards_1[1:],\
                                            sub_observations_2[1:],sub_actions_2[1:],sub_d_rewards_2[1:]):
        iw = f_importance_weights(ob_1,ac_1)
        grad_diff = [sum(x) for x in zip(grad_diff,f_compute_grad_diff(ob_1,ac_1,rw_1,iw))]
        u_est = [sum(x) for x in zip(u_est,f_compute_grad(ob_2,ac_2,rw_2))]

    grad_diff = [x/len(sub_observations_1) for x in grad_diff]
    u_est = [x/len(sub_observations_2) for x in u_est]
    v_est = [beta*(v + grad_d) + (1-beta) * u for v,grad_d,u in zip(v_est,grad_diff,u_est)]

    return v_est