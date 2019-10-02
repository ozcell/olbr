import numpy as np
import scipy as sc
import time

import torch as K

from olbr.utils import get_params as get_params, running_mean, get_exp_params
from olbr.main import init, run
from olbr.main_ppo import init as init_ppo
from olbr.main_ppo import run as run_ppo
import matplotlib.pyplot as plt

import os
import pickle
import sys

filepath='/jmain01/home/JAD022/grm01/oxk28-grm01/Dropbox/Jupyter/notebooks/Reinforcement_Learning/'
os.chdir(filepath)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

exp_config = get_exp_params(sys.argv[1:])

if exp_config['env'] == 'Push':
    env_name = 'FetchPushMulti-v1'
elif exp_config['env'] == 'PnP':
    env_name = 'FetchPickAndPlaceMulti-v1'

if exp_config['multiseed'] == 'True':
    multiseed = True
    n_envs = 38
    from olbr.main import init, run
elif exp_config['multiseed'] == 'False':
    multiseed = False
    n_envs = 1
    from olbr.main import init, run

masked_with_r = exp_config['masked_with_r']
if exp_config['use_her'] == 'True':
    use_her = True
else:
    use_her = False

for i_exp in range(0,int(exp_config['n_exp'])):
    if exp_config['obj_rew'] == 'True':
    ####################### loading object ###########################

        model_name = 'DDPG_BD'
        exp_args=['--env_id', env_name,
                '--exp_id', model_name + '_fooobj_' + str(0),
                '--random_seed', str(0), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', '0.98',
                '--n_episodes', '20',
                '--n_cycles', '50',
                '--n_rollouts', '38',
                '--n_test_rollouts', '38',
                '--n_envs', str(n_envs),
                '--n_batches', '40',
                '--batch_size', '256',
                '--n_bd_batches', '400',
                '--obj_action_type', '012',
                '--max_nb_objects', '1',
                '--observe_obj_grp', 'False',
                '--rob_policy', '01',
                ]

        config = get_params(args=exp_args)
        model, experiment_args = init(config, agent='object', her=True, 
                                    object_Qfunc=None, 
                                    backward_dyn=None,
                                    object_policy=None)
        env, memory, noise, config, normalizer, agent_id = experiment_args

        #loading the object model
        if exp_config['env'] == 'Push':
            path = './models/obj/obj_push_xyz/'
        elif exp_config['env'] == 'PnP':
            path = './models/obj/obj_pnp_xyz/'

        model.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
        model.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
        model.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
        with open(path + 'normalizer.pkl', 'rb') as file:
            normalizer = pickle.load(file)

        experiment_args = (env, memory, noise, config, normalizer, agent_id)
        
        obj_rew = True
        object_Qfunc = model.critics[0]
        backward_dyn = model.backward
        object_policy = model.actors[0]  
    ####################### loading object ###########################
    elif exp_config['obj_rew'] == 'False':
        obj_rew = False
        object_Qfunc = None
        backward_dyn = None
        object_policy = None  

    ####################### training robot ###########################  
    if exp_config['rob_model'] == 'DDPG':
        model_name = 'DDPG_BD'
        exp_args2=['--env_id', env_name,
                '--exp_id', model_name + '_foorob_' + str(i_exp),
                '--random_seed', str(i_exp), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', '0.98',
                '--n_episodes', '50',
                '--n_cycles', '50',
                '--n_rollouts', '38',
                '--n_test_rollouts', '380',
                '--n_envs', str(n_envs),
                '--n_batches', '40',
                '--batch_size', '256',
                '--n_bd_batches', '400',
                '--obj_action_type', '012',
                '--max_nb_objects', '1',
                '--observe_obj_grp', 'False',
                '--rob_policy', '01',
                '--masked_with_r', masked_with_r,
                ]

        config2 = get_params(args=exp_args2)
        model2, experiment_args2 = init(config2, agent='robot', her=use_her, 
                                        object_Qfunc=object_Qfunc, 
                                        backward_dyn=backward_dyn,
                                        object_policy=object_policy
                                    )
        env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2
        if obj_rew:
            normalizer2[1] = normalizer[1]
        experiment_args2 = (env2, memory2, noise2, config2, normalizer2, agent_id2)

        monitor2 = run(model2, experiment_args2, train=True)

    elif exp_config['rob_model'] == 'PPO':
        model_name = 'PPO_BD'
        exp_args2=['--env_id', env_name,
                '--exp_id', model_name + '_foorob_' + str(i_exp),
                '--random_seed', str(i_exp), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', '0.98',
                '--n_episodes', '50',
                '--n_cycles', '50',
                '--n_rollouts', '38',
                '--n_batches', '4',
                '--batch_size', '256',
                '--reward_normalization', 'False', 
                '--ai_object_rate', '0.0',
                '--obj_action_type', '012',
                '--max_nb_objects', '1',
                '--observe_obj_grp', 'True',
                '--masked_with_r', masked_with_r,
                '--n_test_rollouts', '100',
                '--plcy_lr', '3e-4',
                '--crtc_lr', '3e-4',
                '--ppo_epoch', '3',
                '--entropy_coef', '0.00',
                '--clip_param', '0.1',
                '--use_gae', "False"
        ]

        config2 = get_params(args=exp_args2)
        model2, experiment_args2 = init_ppo(config2, agent='robot', her=False, 
                                        object_Qfunc=object_Qfunc, 
                                        backward_dyn=backward_dyn,
                                        object_policy=object_policy
                                    )
        env2, memory2, noise2, config2, normalizer2, agent_id2 = experiment_args2
        normalizer2[1] = normalizer[1]
        experiment_args2 = (env2, memory2, noise2, config2, normalizer2, agent_id2)

        monitor2 = run_ppo(model2, experiment_args2, train=True)

    rob_name = exp_config['env']
    if exp_config['rob_model'] == 'PPO':
        if obj_rew:
            rob_name = rob_name + '_v4_PPO_'
        else:
            rob_name = rob_name + '_PPO_'
    elif exp_config['rob_model'] == 'DDPG':
        if obj_rew:
            if use_her:
                rob_name = rob_name + '_v4_HER_'
            else:
                rob_name = rob_name + '_v4_DDPG_'
        else:
            if use_her:
                rob_name = rob_name + '_HER_'
            else:
                rob_name = rob_name + '_DDPG_'

    rob_name = rob_name + 'Norm_Slide_Clipped_Both_'
    if masked_with_r:
        rob_name = rob_name + 'Masked_PlusR'
    else:
        rob_name = rob_name + 'PlusR'

    if multiseed:
        rob_name = rob_name + '_Multiseed'
    else:
        rob_name = rob_name + '_Singleseed'

    path = './models/_recent/rob_model_' + rob_name + '_' + str(i_exp)
    try:  
        os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s" % path)

    K.save(model2.critics[0].state_dict(), path + '/robot_Qfunc.pt')
    K.save(model2.actors[0].state_dict(), path + '/robot_policy.pt')
    if obj_rew:
        K.save(model2.object_Qfunc.state_dict(), path + '/object_Qfunc.pt')
        K.save(model2.backward.state_dict(), path + '/backward_dyn.pt')
        K.save(model2.object_policy.state_dict(), path + '/object_policy.pt')
    
    with open(path + '/normalizer.pkl', 'wb') as file:
        pickle.dump(normalizer2, file)

    path = './monitors/_recent/monitor_' + rob_name  + '_' + str(i_exp) + '.npy'
    np.save(path, monitor2)

