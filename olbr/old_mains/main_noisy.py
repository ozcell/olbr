import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from olbr.algorithms.ddpg import DDPG_BD
from olbr.algorithms.maddpg import MADDPG_BD
from olbr.experience import Normalizer
from olbr.exploration import Noise
from olbr.utils import Saver, Summarizer, get_params, running_mean
from olbr.agents.basic import Actor 
from olbr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, backward_dyn=None, object_policy=None, reward_fun=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']

    if (ENV_NAME == 'FetchStackMulti-v1') or (ENV_NAME == 'FetchPushMulti-v1') or (ENV_NAME == 'FetchPickAndPlaceMulti-v1'):
        env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], obj_action_type=config['obj_action_type'], observe_obj_grp=config['observe_obj_grp'])
    else:
        env = gym.make(ENV_NAME)

    def her_reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
    
    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)

    if config['obj_action_type'] == 'all':
        n_actions = config['max_nb_objects'] * 7 + 4
    elif config['obj_action_type'] == 'slide_only':
        n_actions = config['max_nb_objects'] * 3 + 4
    elif config['obj_action_type'] == 'rotation_only':
        n_actions = config['max_nb_objects'] * 4 + 4

    observation_space = env.observation_space.spaces['observation'].shape[1] + env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-4,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))

    GAMMA = config['gamma'] 
    TAU = config['tau'] 
    ACTOR_LR = config['plcy_lr'] 
    CRITIC_LR = config['crtc_lr'] 

    MEM_SIZE = config['buffer_length']

    REGULARIZATION = config['regularization']
    NORMALIZED_REWARDS = config['reward_normalization']

    OUT_FUNC = K.tanh 
    if config['agent_alg'] == 'DDPG_BD':
        MODEL = DDPG_BD
        from olbr.replay_buffer import ReplayBuffer
        from olbr.her_sampler import make_sample_her_transitions
    elif config['agent_alg'] == 'MADDPG_BD':
        MODEL = MADDPG_BD
        from olbr.replay_buffer import ReplayBuffer_v2 as ReplayBuffer
        from olbr.her_sampler import make_sample_her_transitions_v2 as make_sample_her_transitions

    #exploration initialization
    if agent == 'robot':
        agent_id = 0
        noise = Noise(action_space[0].shape[0], sigma=0.2, eps=0.3)
    elif agent == 'object':
        agent_id = 1
        #noise = Noise(action_space[1].shape[0], sigma=0.05, eps=0.1)
        noise = Noise(action_space[1].shape[0], sigma=0.2, eps=0.3)

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                  regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                  agent_id=agent_id, object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
                  object_policy=object_policy, reward_fun=reward_fun)
    normalizer = [Normalizer(), Normalizer()]

    #memory initilization  
    if her:
        sample_her_transitions = make_sample_her_transitions('future', 4, her_reward_fun)
    else:
        sample_her_transitions = make_sample_her_transitions('none', 4, her_reward_fun)

    buffer_shapes = {
        'o' : (env._max_episode_steps, env.observation_space.spaces['observation'].shape[1]*2),
        'ag' : (env._max_episode_steps, env.observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (env._max_episode_steps, env.observation_space.spaces['desired_goal'].shape[0]),
        'u' : (env._max_episode_steps-1, action_space[2].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, env._max_episode_steps, sample_her_transitions)

    experiment_args = (env, memory, noise, config, normalizer, agent_id)

    print('train BD x10')
          
    return model, experiment_args

def rollout(env, model, noise, normalizer=None, render=False, agent_id=0, ai_object=False):
    trajectories = []
    
    # monitoring variables
    episode_reward = 0
    frames = []
    
    env.env.ai_object = True if agent_id==1 else ai_object
    state_all = env.reset()

    for i_agent in range(2):
        trajectories.append([])
    
    for i_step in range(env._max_episode_steps):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        # Observation normalization
        obs_goal = []
        for i_agent in range(2):
            obs_goal.append(K.cat([obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])

        action = model.select_action(obs_goal[agent_id], noise).cpu().numpy().squeeze(0)

        if agent_id == 0:
            action_to_env = np.zeros_like(env.action_space.sample())
            action_to_env[0:action.shape[0]] = action
            if ai_object:
                obj_noise = Noise(3, sigma=0.2, eps=0.3)
                
                obj_action = model.get_obj_action(obs_goal[1]).cpu().numpy().squeeze(0)
                obj_action = obj_noise.get_noisy_action(obj_action).clip(-1.0, 1.0)
                action_to_env[action.shape[0]::] = obj_action

        else:
            action_to_env = env.action_space.sample()
            action_to_env[-action.shape[0]::] = action

        next_state_all, reward, done, info = env.step(action_to_env)
        reward = K.tensor(reward, dtype=dtype).view(1,1)

        next_obs = [K.tensor(next_obs, dtype=K.float32).unsqueeze(0) for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        for i_agent in range(2):
            next_obs_goal.append(K.cat([next_obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])

        # for monitoring
        if agent_id == 0:
            episode_reward += (model.get_obj_reward(obs_goal[1], next_obs_goal[1]) + reward)
        else:
            episode_reward += reward

        for i_agent in range(2):
            state = {
                'observation'   : state_all['observation'][i_agent],
                'achieved_goal' : state_all['achieved_goal'],
                'desired_goal'  : state_all['desired_goal']   
                }
            next_state = {
                'observation'   : next_state_all['observation'][i_agent],
                'achieved_goal' : next_state_all['achieved_goal'],
                'desired_goal'  : next_state_all['desired_goal']    
                }
            
            trajectories[i_agent].append((state.copy(), action_to_env, reward, next_state.copy(), done))

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    obs, ags, goals, acts = [], [], [], []

    for trajectory in trajectories:
        obs.append([])
        ags.append([])
        goals.append([])
        acts.append([])
        for i_step in range(env._max_episode_steps):
            obs[-1].append(trajectory[i_step][0]['observation'])
            ags[-1].append(trajectory[i_step][0]['achieved_goal'])
            goals[-1].append(trajectory[i_step][0]['desired_goal'])
            if (i_step < env._max_episode_steps - 1): 
                acts[-1].append(trajectory[i_step][1])

    trajectories = {
        'o'    : np.concatenate(obs,axis=1)[np.newaxis,:],
        'ag'   : np.asarray(ags)[0:1,],
        'g'    : np.asarray(goals)[0:1,],
        'u'    : np.asarray(acts)[0:1,],
    }

    return trajectories, episode_reward, info['is_success'], frames

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, memory, noise, config, normalizer, agent_id = experiment_args
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_ROLLOUTS = config['n_rollouts']
    N_BATCHES = config['n_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    critic_losses = []
    actor_losses = []
    backward_losses = []
        
    for i_episode in range(N_EPISODES):
        
        episode_time_start = time.time()
        if train:
            for i_cycle in range(N_CYCLES):
                
                for i_rollout in range(N_ROLLOUTS):
                    # Initialize the environment and state
                    ai_object = 1 if np.random.rand() < config['ai_object_rate']  else 0
                    trajectories, _, _, _ = rollout(env, model, noise, normalizer, render=(i_rollout==N_ROLLOUTS-1), agent_id=agent_id, ai_object=ai_object)
                    memory.store_episode(trajectories.copy())   
              
                for i_batch in range(N_BATCHES):  
                    model.to_cuda()

                    batch = memory.sample(BATCH_SIZE)
                    critic_loss, actor_loss = model.update_parameters(batch, normalizer)

                    if i_batch == N_BATCHES - 1:
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                model.update_target()

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(N_TEST_ROLLOUTS):
            # Initialize the environment and state
            render = config['render'] > 0 and i_episode % config['render'] == 0
            _, episode_reward, success, _ = rollout(env, model, noise=False, normalizer=normalizer, render=render, agent_id=agent_id, ai_object=False)
                
            episode_reward_cycle.append(episode_reward)
            episode_succeess_cycle.append(success)
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(np.mean(episode_reward_cycle))
        episode_success_all.append(np.mean(episode_succeess_cycle))
        plot_durations(np.asarray(episode_reward_all), np.asarray(episode_success_all))
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Tensorboard port: {}'.format(config['port']))
                print('  | Episode total reward: {}'.format(episode_reward))
                print('  | Running mean of total reward: {}'.format(episode_reward_all[-1]))
                print('  | Success rate: {}'.format(episode_success_all[-1]))
                #print('  | Running mean of total reward: {}'.format(running_mean(episode_reward_all)[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
            
    # <-- end loop: i_episode

    if train and agent_id==1:
        print('Training Backward Model')
        model.to_cuda()
        for _ in range(N_EPISODES*N_CYCLES*10):
            for i_batch in range(N_BATCHES):
                batch = memory.sample(BATCH_SIZE)
                backward_loss = model.update_backward(batch, normalizer)  
                if i_batch == N_BATCHES - 1:
                    backward_losses.append(backward_loss)
        plot_durations(np.asarray(backward_losses), np.asarray(backward_losses))

    if train:
        print('Training completed')
    else:
        print('Test completed')

    
    return episode_reward_all, episode_success_all

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(p, r):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(p)
    plt.plot(r)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())