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
from olbr.experience import Normalizer, RunningMean
from olbr.exploration import Noise
from olbr.utils import Saver, Summarizer, get_params
from olbr.agents.basic import Actor 
from olbr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, 
                                           backward_dyn=None, 
                                           object_policy=None, 
                                           reward_fun=None,
                                           rnd_models=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']
    N_ENVS = config['n_envs']

    env = []
    if 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                          obj_action_type=config['obj_action_type'], 
                                          observe_obj_grp=config['observe_obj_grp'],
                                          obj_range=config['obj_range']))
        n_rob_actions = 4
        n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + n_rob_actions
    elif 'HandManipulate' in ENV_NAME and 'Multi' in ENV_NAME:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME, obj_action_type=config['obj_action_type']))
        n_rob_actions = 20
        n_actions = 1 * len(config['obj_action_type']) + n_rob_actions
    else:
        for i_env in range(N_ENVS):
            env.append(gym.make(ENV_NAME))

    def her_reward_fun(ag_2, g, info):  # vectorized
        return env[0].compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    for i_env in range(N_ENVS):
        env[i_env].seed(SEED+10*i_env)
    K.manual_seed(SEED)
    np.random.seed(SEED)

    observation_space = env[0].observation_space.spaces['observation'].shape[1] + env[0].observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-n_rob_actions,), dtype='float32'),
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
    agent_id = 0
    noise = Noise(action_space[0].shape[0], sigma=0.2, eps=0.3)
    env[0]._max_episode_steps *= config['max_nb_objects']

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                  regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                  agent_id=agent_id, object_Qfunc=object_Qfunc, backward_dyn=backward_dyn, 
                  object_policy=object_policy, reward_fun=reward_fun, masked_with_r=config['masked_with_r'],
                  rnd_models=rnd_models, pred_th=config['pred_th'])
    normalizer = [Normalizer(), Normalizer()]

    #memory initilization  
    if her:
        sample_her_transitions = make_sample_her_transitions('future', 4, her_reward_fun)
    else:
        sample_her_transitions = make_sample_her_transitions('none', 4, her_reward_fun)

    buffer_shapes = {
        'o' : (env[0]._max_episode_steps, env[0].observation_space.spaces['observation'].shape[1]*2),
        'ag' : (env[0]._max_episode_steps, env[0].observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (env[0]._max_episode_steps, env[0].observation_space.spaces['desired_goal'].shape[0]),
        'u' : (env[0]._max_episode_steps-1, action_space[2].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, env[0]._max_episode_steps, sample_her_transitions)

    running_rintr_mean = RunningMean()

    experiment_args = (env, memory, noise, config, normalizer, running_rintr_mean)

    print('keep updating the normaliser')
          
    return model, experiment_args

ALL_COUNT = 0.
NC_COUNT = 0.

def rollout(env, model, noise, i_env, normalizer=None, render=False, running_rintr_mean=None):
    trajectories = []
    for i_agent in range(2):
        trajectories.append([])
    
    # monitoring variables
    episode_reward = 0
    frames = []
    
    env[i_env].env.ai_object = False
    env[i_env].env.deactivate_ai_object() 
    state_all = env[i_env].reset()

    for i_step in range(env[0]._max_episode_steps):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)

        # Observation normalization
        obs_goal = []
        for i_agent in range(2):
            obs_goal.append(K.cat([obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                if i_agent == 0:
                    obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])
                else:
                    obs_goal[i_agent] = normalizer[i_agent].preprocess_with_update(obs_goal[i_agent])

        action = model.select_action(obs_goal[0], noise).cpu().numpy().squeeze(0)

        action_to_env = np.zeros_like(env[0].action_space.sample())
        action_to_env[0:action.shape[0]] = action

        next_state_all, reward, done, info = env[i_env].step(action_to_env)
        reward = K.tensor(reward, dtype=dtype).view(1,1)

        next_obs = [K.tensor(next_obs, dtype=K.float32).unsqueeze(0) for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        for i_agent in range(2):
            next_obs_goal.append(K.cat([next_obs[i_agent], goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])

        pred_error = model.get_pred_error(next_obs_goal[1]).cpu().numpy().squeeze(0)

        global ALL_COUNT
        global NC_COUNT

        ALL_COUNT += 1.
        # for monitoring
        if model.object_Qfunc is None:            
            episode_reward += reward
        else:
            if pred_error > model.pred_th:
                NC_COUNT +=1.
                if running_rintr_mean is not None:
                    r_intr = K.tensor(running_rintr_mean.get_stats(), dtype=obs_goal[1].dtype, device=obs_goal[1].device)
                else:
                    r_intr = -0.5
            else:
                r_intr = model.get_obj_reward(obs_goal[1], next_obs_goal[1])
                if running_rintr_mean is not None:
                    running_rintr_mean.update_stats(r_intr.cpu().numpy().squeeze(0))

            if model.masked_with_r:
                episode_reward += (r_intr * K.abs(reward) + reward)
            else:
                episode_reward += (r_intr + reward)

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
            frames.append(env[i_env].render(mode='rgb_array')[0])

    obs, ags, goals, acts = [], [], [], []

    for trajectory in trajectories:
        obs.append([])
        ags.append([])
        goals.append([])
        acts.append([])
        for i_step in range(env[0]._max_episode_steps):
            obs[-1].append(trajectory[i_step][0]['observation'])
            ags[-1].append(trajectory[i_step][0]['achieved_goal'])
            goals[-1].append(trajectory[i_step][0]['desired_goal'])
            if (i_step < env[0]._max_episode_steps - 1): 
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

    env, memory, noise, config, normalizer, running_rintr_mean = experiment_args
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_ROLLOUTS = config['n_rollouts']
    N_BATCHES = config['n_batches']
    N_BD_BATCHES = config['n_bd_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    episode_reward_mean = []
    episode_success_mean = []
    critic_losses = []
    actor_losses = []
        
    for i_episode in range(N_EPISODES):
        
        episode_time_start = time.time()
        if train:
            for i_cycle in range(N_CYCLES):
                
                for i_rollout in range(N_ROLLOUTS):
                    # Initialize the environment and state
                    i_env = i_rollout % config['n_envs']
                    render = config['render'] > 0 and i_rollout==0
                    trajectories, _, _, _ = rollout(env, model, noise, i_env, normalizer, render=render, running_rintr_mean=running_rintr_mean)
                    memory.store_episode(trajectories.copy())   
              
                for i_batch in range(N_BATCHES):  
                    model.to_cuda()

                    batch = memory.sample(BATCH_SIZE)
                    critic_loss, actor_loss = model.update_parameters(batch, normalizer, running_rintr_mean)
                    if i_batch == N_BATCHES - 1:
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                model.update_target()

                if NC_COUNT > 0:
                    print(NC_COUNT/ALL_COUNT)
                    print(running_rintr_mean.get_stats())

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        for i_rollout in range(N_TEST_ROLLOUTS):
            # Initialize the environment and state
            rollout_per_env = N_TEST_ROLLOUTS // config['n_envs']
            i_env = i_rollout // rollout_per_env
            render = config['render'] > 0 and i_episode % config['render'] == 0 and i_env == 0
            _, episode_reward, success, _ = rollout(env, model, False, i_env, normalizer=normalizer, render=render, running_rintr_mean=running_rintr_mean)
                
            episode_reward_cycle.append(episode_reward.item())
            episode_succeess_cycle.append(success)
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(episode_reward_cycle)
        episode_success_all.append(episode_succeess_cycle)

        episode_reward_mean.append(np.mean(episode_reward_cycle))
        episode_success_mean.append(np.mean(episode_succeess_cycle))
        plot_durations(np.asarray(episode_reward_mean), np.asarray(episode_success_mean))
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Episode total reward: {}'.format(episode_reward))
                print('  | Running mean of total reward: {}'.format(episode_reward_mean[-1]))
                print('  | Success rate: {}'.format(episode_success_mean[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
            
    # <-- end loop: i_episode
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