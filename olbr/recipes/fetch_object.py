import numpy as np
import scipy as sc
import time
import pickle
import os
import sys

import torch as K
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from olbr.utils import get_params as get_params, running_mean, get_exp_params
from olbr.main_rnd import init, run, interfere
from olbr.agents.basic import TrajectoryDyn

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

K.set_num_threads(1)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx]

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

    if 'FetchPushObstacleDoubleGapMulti' in env_name:
        n_episodes = 50
        gamma = 0.9875
    else: 
        n_episodes = 25
        gamma = 0.98

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
    elif env_name == 'FetchPickAndPlaceInsertionMulti-v1':
        path = './ObT_models/obj/pnp_insertion_7d_ep25/'
    print('object trajectory model will be created in')
    print(path)        


    #training object policy
    model_name = 'DDPG_BD'
    exp_args=['--env_id', env_name,
            '--exp_id', model_name + '_foorob_' + str(0),
            '--random_seed', str(0), 
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
            '--rob_policy', '02',
            '--buffer_length', '10000000',
            ]

    config = get_params(args=exp_args)
    model, experiment_args = init(config, agent='object', her=True, 
                                object_Qfunc=None, 
                                backward_dyn=None,
                                object_policy=None)
    env, memory, noise, config, normalizer, agent_id = experiment_args
    monitor = run(model, experiment_args, train=True)

    try:  
        os.makedirs(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s" % path)

    K.save(model.critics[0].state_dict(), path + 'object_Qfunc.pt')
    K.save(model.actors[0].state_dict(), path + 'object_policy.pt')
    K.save(model.backward.state_dict(), path + 'backward_dyn.pt')
    with open(path + 'normalizer.pkl', 'wb') as file:
        pickle.dump(normalizer, file)  
        
    np.save(path + 'monitor', monitor)

    #generating object trajectories
    memory.clear_buffer()
    interfere(model, experiment_args)

    #creating the dataset from trajectories
    time_step = 1
    horizon=n_episodes
    goal_len = memory.buffers['ag'].shape[2]

    ag_in_all = []
    ag_out_all = []
    g_in_all = []
    step_in_all = []
    for i_time in range(time_step,horizon+time_step,time_step):
        ag_in = memory.buffers['ag'][0:memory.current_size,0,0:goal_len]
        if i_time == horizon:
            ag_out = memory.buffers['g'][0:memory.current_size,0,0:goal_len]
        else:
            ag_out = memory.buffers['ag'][0:memory.current_size,i_time,0:goal_len]
        g_in = memory.buffers['g'][0:memory.current_size,0,0:goal_len]
        
        ag_in_all.append(ag_in.reshape(-1,goal_len).astype('float32'))
        ag_out_all.append(ag_out.reshape(-1,goal_len).astype('float32'))
        g_in_all.append(g_in.reshape(-1,goal_len).astype('float32'))
        step_in_all.append((np.ones_like(ag_in)[:,0]*i_time).astype('float32'))
        
    ag_in_all = np.asarray(ag_in_all).reshape(-1,goal_len)
    ag_out_all = np.asarray(ag_out_all).reshape(-1,goal_len)
    g_in_all = np.asarray(g_in_all).reshape(-1,goal_len)
    step_in_all = np.asarray(step_in_all).reshape(-1,1)
        
    ag_in_mean = memory.buffers['ag'][0:memory.current_size].mean((0,1))
    ag_in_std = memory.buffers['ag'][0:memory.current_size].std((0,1))
    g_in_mean = memory.buffers['g'][0:memory.current_size].mean((0,1))
    g_in_std = memory.buffers['g'][0:memory.current_size].std((0,1))

    step_in_all_mean = step_in_all.mean(0)
    step_in_all_std = step_in_all.std(0)

    obj_mean = [ag_in_mean, g_in_mean, step_in_all_mean]
    obj_std = [ag_in_std, g_in_std, step_in_all_std]

    with open(path + 'objGoal_mean.pkl', 'wb') as file:
        pickle.dump(obj_mean, file)
    with open(path + 'objGoal_std.pkl', 'wb') as file:
        pickle.dump(obj_std, file)
        
    train_split = 0.7
    train_bound = int(train_split*len(ag_in_all))
    idxs = np.random.permutation(len(ag_in_all))
    train_idxs = idxs[0:train_bound]
    test_idxs = idxs[train_bound::]

    ag_in_all -= ag_in_mean
    ag_in_all /= (ag_in_std + 1e-5)

    g_in_all -= g_in_mean
    g_in_all /= (g_in_std + 1e-5)

    step_in_all -= step_in_all_mean
    step_in_all /= step_in_all_std

    train_data = (ag_in_all[train_idxs], g_in_all[train_idxs], step_in_all[train_idxs], ag_out_all[train_idxs])
    test_data = (ag_in_all[test_idxs], g_in_all[test_idxs], step_in_all[test_idxs], ag_out_all[test_idxs])

    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    #initialising the trajectory model
    model_objtraj = TrajectoryDyn(goal_len).to(device)
    optimizer = optim.Adam(model_objtraj.parameters())

    #training the trajectory model
    nb_traj_epochs = 30
    for epoch in range(nb_traj_epochs): 

        # training
        running_loss = 0.0
        total = 0.

        model_objtraj.train()
        for data in train_loader:
            # get the inputs
            X1, X2, X3, T = data
            X1 = X1.to(device)
            X2 = X2.to(device)
            X3 = X3.to(device)
            T = T.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            Y = model_objtraj(X1, X2, X3)
            loss = F.mse_loss(Y, T)
            loss.backward()
            optimizer.step()

            # print statistics
            total += T.size(0)
            running_loss += loss.item()#*T.size(0)

        #running_loss /= total

        # testing
        correct_Y = 0.
        total_test = 0. 

        model_objtraj.eval()
        with K.no_grad():
            for data in test_loader:
                # get the inputs
                X1, X2, X3, T = data
                X1 = X1.to(device)
                X2 = X2.to(device)
                X3 = X3.to(device)
                T = T.to(device)
                # 
                Y = model_objtraj(X1, X2, X3)

                total_test += T.size(0)
                loss = F.mse_loss(Y, T)
                
                correct_Y += loss.item()
                
            #correct_Y /= total_test

        print('%d, loss: %.3f, acc = %.3f %%'  % (epoch + 1, 
                                                running_loss, 
                                                correct_Y))
    print('Finished Training of Object Trajectory Model')                                            
    K.save(model_objtraj.state_dict(), path + 'objGoal_model.pt')