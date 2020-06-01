import numpy as np
import scipy as sc
import time

import torch as K

from olbr.utils import get_params as get_params, running_mean, get_exp_params
from olbr.main_cher import init, run
from olbr.main_q_v2 import init as init_q
from olbr.main_q_v2 import run as run_q
from olbr.main_q_rnd import init as init_q_rnd
from olbr.main_q_rnd import run as run_q_rnd
import matplotlib.pyplot as plt

import os
import pickle
import sys
import olbr


K.set_num_threads(1)

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

exp_config = get_exp_params(sys.argv[1:])
filepath = exp_config['filepath'] 
os.chdir(filepath)

suffix = ''

if exp_config['env'] == 'PushSide':
     env_name_list = ['FetchPushObstacleSideGapMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PushMiddle':
     env_name_list = ['FetchPushObstacleMiddleGapMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PushDouble':
     env_name_list = ['FetchPushObstacleDoubleGapMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PnPShelf':
     env_name_list = ['FetchPickAndPlaceShelfMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PnPObstacle':
     env_name_list = ['FetchPickAndPlaceObstacleMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PnPHardest':
     env_name_list = ['FetchPickAndPlaceHardestMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PnPInsertion':
     env_name_list = ['FetchPickAndPlaceInsertionMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'PnPNormal':
     env_name_list = ['FetchPickAndPlaceMulti{}-v1'.format(suffix)]
elif exp_config['env'] == 'All':
     env_name_list = ['FetchPushObstacleSideGapMulti{}-v1'.format(suffix), 
                      'FetchPushObstacleMiddleGapMulti{}-v1'.format(suffix), 
                      'FetchPushObstacleDoubleGapMulti{}-v1'.format(suffix), 
                      'FetchPickAndPlaceShelfMulti{}-v1'.format(suffix), 
                      'FetchPickAndPlaceObstacleMulti{}-v1'.format(suffix),
                      'FetchPickAndPlaceHardestMulti{}-v1'.format(suffix),
                      'FetchPickAndPlaceInsertionMulti{}-v1'.format(suffix),
                      'FetchPickAndPlaceMulti{}-v1'.format(suffix)
                      ]

for env_name in env_name_list:

    if exp_config['use_cher'] == 'True':
        use_cher = True
        print("training with CHER")
    else:
        use_cher = False
        print("training without CHER")

    if 'FetchPushObstacleDoubleGapMulti' in env_name:
        n_episodes = 200
        gamma = 0.9875
    else: 
        n_episodes = 100
        gamma = 0.98

    for i_exp in range(int(exp_config['start_n_exp']), int(exp_config['n_exp'])):
        obj_rew = False
        object_Qfunc = None
        object_policy = None  
        backward_dyn = None
        print("training without object based rewards")
        init_2 = init
        run_2 = run

        ####################### training robot ###########################  
        model_name = 'DDPG_BD'
        exp_args2=['--env_id', env_name,
                '--exp_id', model_name + '_foorob_' + str(i_exp),
                '--random_seed', str(i_exp), 
                '--agent_alg', model_name,
                '--verbose', '2',
                '--render', '0',
                '--gamma', str(gamma),
                '--n_episodes', str(n_episodes),
                '--n_cycles', '50',
                '--n_rollouts', '38',
                '--n_test_rollouts', '380',
                '--n_envs', '38',
                '--n_batches', '40',
                '--batch_size', '4864',
                '--obj_action_type', '0123456',
                '--max_nb_objects', '1',
                '--observe_obj_grp', 'False',
                ]

        config2 = get_params(args=exp_args2)
        model2, experiment_args2 = init_2(config2, agent='robot', her=use_cher, 
                                        object_Qfunc=object_Qfunc,
                                        object_policy=object_policy,
                                        backward_dyn=backward_dyn,
                                    )            
        env2, memory2, noise2, config2, normalizer2, running_rintr_mean2 = experiment_args2
        experiment_args2 = (env2, memory2, noise2, config2, normalizer2, running_rintr_mean2)

        monitor2, bestmodel = run_2(model2, experiment_args2, train=True)

        rob_name = env_name
        if use_cher:
            rob_name = rob_name + '_CHER_'
        else:
            rob_name = rob_name + '_'

        path = './ObT_models/batch1/' + rob_name + str(i_exp)
        try:  
            os.makedirs(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s" % path)

        K.save(model2.critics[0].state_dict(), path + '/robot_Qfunc.pt')
        K.save(model2.actors[0].state_dict(), path + '/robot_policy.pt')

        K.save(bestmodel[0], path + '/robot_Qfunc_best.pt')
        K.save(bestmodel[1], path + '/robot_policy_best.pt')
        
        with open(path + '/normalizer.pkl', 'wb') as file:
            pickle.dump(normalizer2, file)

        with open(path + '/normalizer_best.pkl', 'wb') as file:
            pickle.dump(bestmodel[2], file)

        path = './ObT_models/batch1/monitor_' + rob_name + str(i_exp) + '.npy'
        np.save(path, monitor2)

