import numpy as np
import scipy as sc
import time

import torch as K

from olbr.utils import get_params as get_params, running_mean, get_exp_params
from olbr.main import init, run
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

if exp_config['shaped'] == 'True':
    use_dist = True
else:
    use_dist = False

suffix = 'Dense' if use_dist else ''

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
elif exp_config['env'] == 'All':
     env_name_list = ['FetchPushObstacleSideGapMulti{}-v1'.format(suffix), 
                      'FetchPushObstacleMiddleGapMulti{}-v1'.format(suffix), 
                      'FetchPushObstacleDoubleGapMulti{}-v1'.format(suffix), 
                      'FetchPickAndPlaceShelfMulti{}-v1'.format(suffix), 
                      'FetchPickAndPlaceObstacleMulti{}-v1'.format(suffix),
                      'FetchPickAndPlaceHardestMulti{}-v1'.format(suffix)]

for env_name in env_name_list:

    if exp_config['use_her'] == 'True':
        use_her = True
        print("training with HER")
    else:
        use_her = False
        print("training without HER")

    if exp_config['use_rnd'] == 'True':
        use_rnd = True
    else:
        use_rnd = False

    if env_name == 'FetchPushObstacleDoubleGapMulti-v1':
        n_episodes = 200
        gamma = 0.9875
    else: 
        n_episodes = 100
        gamma = 0.98

    for i_exp in range(int(exp_config['start_n_exp']), int(exp_config['n_exp'])):
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
                    '--n_envs', '1',
                    '--n_batches', '40',
                    '--batch_size', '4864',
                    '--obj_action_type', '0123456',
                    '--max_nb_objects', '1',
                    '--observe_obj_grp', 'False',
                    '--rob_policy', '02',
                    ]                   

            config = get_params(args=exp_args)
            model, experiment_args = init(config, agent='object', her=True, 
                                        object_Qfunc=None, 
                                        backward_dyn=None,
                                        object_policy=None)
            env, memory, noise, config, normalizer, agent_id = experiment_args

            #loading the object model
            if env_name == 'FetchPushObstacleSideGapMulti-v1':
                path = './ObT_models/obj/push_side_7d_ep25/'
            elif env_name == 'FetchPushObstacleMiddleGapMulti-v1':
                path = './ObT_models/obj/push_middle_7d_ep25/'
            elif env_name == 'FetchPushObstacleDoubleGapMulti-v1':
                path = './ObT_models/obj/push_double_7d_ep50/'
            elif env_name == 'FetchPickAndPlaceShelfMulti-v1':
                path = './ObT_models/obj/pnp_shelf_7d_ep25/'
            elif env_name == 'FetchPickAndPlaceObstacleMulti-v1':
                path = './ObT_models/obj/pnp_obstacle_7d_ep25/'        
            elif env_name == 'FetchPickAndPlaceHardestMulti-v1':
                path = './ObT_models/obj/pnp_hardest_7d_ep25/'

            print('loading object model')
            print(path)
            model.critics[0].load_state_dict(K.load(path + 'object_Qfunc.pt'))
            model.actors[0].load_state_dict(K.load(path + 'object_policy.pt'))
            model.backward.load_state_dict(K.load(path + 'backward_dyn.pt'))
            with open(path + 'normalizer.pkl', 'rb') as file:
                normalizer = pickle.load(file)

            experiment_args = (env, memory, noise, config, normalizer, agent_id)
            
            obj_rew = True
            object_Qfunc = (model.critics[0],)
            object_policy = (model.actors[0],)  
            object_inverse = (model.backward,)
            init_2 = init_q   
            run_2 = run_q
            print("training with object based rewards")
        ####################### loading object ###########################
        elif exp_config['obj_rew'] == 'False':
            obj_rew = False
            object_Qfunc = None
            object_policy = None  
            backward_dyn = None
            print("training without object based rewards")
            if use_rnd:
                init_2 = init_q_rnd
                run_2 = run_q_rnd
                print("training with RND rewards")
            else:
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
        if obj_rew:
            model2, experiment_args2 = init_2(config2, agent='robot', her=use_her, 
                                            object_Qfunc=object_Qfunc,
                                            object_policy=object_policy,
                                            object_inverse=object_inverse,
                                        )
        else:
            model2, experiment_args2 = init_2(config2, agent='robot', her=use_her, 
                                            object_Qfunc=object_Qfunc,
                                            object_policy=object_policy,
                                            backward_dyn=backward_dyn,
                                        )            
        env2, memory2, noise2, config2, normalizer2, running_rintr_mean2 = experiment_args2
        if obj_rew:
            normalizer2[1] = normalizer[1]
        experiment_args2 = (env2, memory2, noise2, config2, normalizer2, running_rintr_mean2)

        monitor2, bestmodel = run_2(model2, experiment_args2, train=True)

        rob_name = env_name
        if obj_rew:
            if use_her:
                rob_name = rob_name + '_SLDR_HER_'
            else:
                rob_name = rob_name + '_SLDR_'
        else:
            if use_her:
                rob_name = rob_name + '_HER_'
            else:
                rob_name = rob_name + '_'


        if use_rnd:
            rob_name = rob_name + 'RND_'

        if use_dist:
            rob_name = rob_name + 'DIST_'

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

        if obj_rew:
            K.save(model2.object_Qfunc[0].state_dict(), path + '/object_Qfunc.pt')
            K.save(model2.object_inverse[0].state_dict(), path + '/backward_dyn.pt')
            K.save(model2.object_policy[0].state_dict(), path + '/object_policy.pt')
        
        with open(path + '/normalizer.pkl', 'wb') as file:
            pickle.dump(normalizer2, file)

        with open(path + '/normalizer_best.pkl', 'wb') as file:
            pickle.dump(bestmodel[2], file)

        path = './ObT_models/batch1/monitor_' + rob_name + str(i_exp) + '.npy'
        np.save(path, monitor2)

