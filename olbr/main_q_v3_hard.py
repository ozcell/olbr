import numpy as np
import scipy as sc
import time
import imageio
import copy

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym_wmgds as gym

from olbr.algorithms.ddpg_q_v2 import DDPG_BD
from olbr.experience import Normalizer, RunningMean
from olbr.exploration import Noise
from olbr.utils import Saver, Summarizer, get_params
from olbr.agents.basic import Actor 
from olbr.agents.basic import Critic

import pdb

import matplotlib
import matplotlib.pyplot as plt

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from gym_wmgds.envs.robotics import rotations

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

def init(config, agent='robot', her=False, object_Qfunc=None, 
                                           object_inverse=None, 
                                           object_policy=None, 
                                           reward_fun=None,
                                           obj_traj=None,
                                           obj_mean=None,
                                           obj_std=None):
        
    #hyperparameters
    ENV_NAME = config['env_id'] 
    SEED = config['random_seed']
    N_ENVS = config['n_envs']

    def make_env(env_id, i_env, env_type='Fetch'):
        def _f():
            if env_type == 'Fetch':
                env = gym.make(env_id, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
            elif env_type == 'Hand':
                env = gym.make(env_id, obj_action_type=config['obj_action_type'])
            elif env_type == 'Others':
                env = gym.make(env_id)
            
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
            env.seed(SEED+10*i_env)
            return env
        return _f    

    if 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME and 'Hard' in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(N_ENVS)])
        if 'Dense' in ENV_NAME:
            envs_test = SubprocVecEnv([make_env('FetchPickAndPlaceMultiDense-v1', i_env, 'Fetch') for i_env in range(N_ENVS)])
        else:
            envs_test = SubprocVecEnv([make_env('FetchPickAndPlaceMulti-v1', i_env, 'Fetch') for i_env in range(N_ENVS)])
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(1)])
        n_rob_actions = 4
        n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + n_rob_actions
    elif 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME and 'Flex' not in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(N_ENVS)])
        envs_test = None
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(1)])
        n_rob_actions = 4
        n_actions = config['max_nb_objects'] * len(config['obj_action_type']) + n_rob_actions
    elif 'Fetch' in ENV_NAME and 'Multi' in ENV_NAME and 'Flex' in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, n_objects=config['max_nb_objects'], 
                                    obj_action_type=config['obj_action_type'], 
                                    observe_obj_grp=config['observe_obj_grp'],
                                    obj_range=config['obj_range'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(N_ENVS)])
        envs_test = None
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Fetch') for i_env in range(1)])
        n_rob_actions = 4
        n_actions = 2 * len(config['obj_action_type']) + n_rob_actions
    elif 'HandManipulate' in ENV_NAME and 'Multi' in ENV_NAME:
        dummy_env = gym.make(ENV_NAME, obj_action_type=config['obj_action_type'])
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Hand') for i_env in range(N_ENVS)])
        envs_test = None
        envs_render = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Hand') for i_env in range(1)])
        n_rob_actions = 20
        n_actions = 1 * len(config['obj_action_type']) + n_rob_actions
    else:
        dummy_env = gym.make(ENV_NAME)
        envs = SubprocVecEnv([make_env(ENV_NAME, i_env, 'Others') for i_env in range(N_ENVS)])
        envs_test = None
        envs_render = None

    def her_reward_fun(ag_2, g, info):  # vectorized
        return dummy_env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    K.manual_seed(SEED)
    np.random.seed(SEED)

    observation_space = dummy_env.observation_space.spaces['observation'].shape[1] + dummy_env.observation_space.spaces['desired_goal'].shape[0]
    action_space = (gym.spaces.Box(-1., 1., shape=(n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions-n_rob_actions,), dtype='float32'),
                    gym.spaces.Box(-1., 1., shape=(n_actions,), dtype='float32'))

    GAMMA = config['gamma'] 
    clip_Q_neg = config['clip_Q_neg'] if config['clip_Q_neg'] < 0 else None
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
    config['episode_length'] = dummy_env._max_episode_steps
    config['observation_space'] = dummy_env.observation_space

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(observation_space, action_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC, discrete=False, 
                  regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS,
                  agent_id=agent_id, object_Qfunc=object_Qfunc, object_inverse=object_inverse, 
                  object_policy=object_policy, reward_fun=reward_fun, clip_Q_neg=clip_Q_neg,
                  )

    normalizer = []
    for _ in range(len(model.object_Qfunc)+1):
        normalizer.append(Normalizer())

    model.n_objects = config['max_nb_objects']

    class NormalizerObj(object):
        def __init__(self, mean, std, epsilon=1e-5):
            self.mean = mean
            self.std = std
            self.epsilon = epsilon

        def process(self, achieved, desired, step):
            achieved_out = achieved - K.tensor(self.mean[0], dtype=achieved.dtype, device=achieved.device)
            achieved_out /= (K.tensor(self.std[0], dtype=achieved.dtype, device=achieved.device) + \
                             K.tensor(self.epsilon, dtype=achieved.dtype, device=achieved.device))

            desired_out = desired - K.tensor(self.mean[1], dtype=desired.dtype, device=desired.device)
            desired_out /= (K.tensor(self.std[1], dtype=desired.dtype, device=desired.device) + \
                            K.tensor(self.epsilon, dtype=desired.dtype, device=desired.device))

            step_out = step - K.tensor(self.mean[2], dtype=desired.dtype, device=desired.device)
            step_out /= K.tensor(self.std[2], dtype=desired.dtype, device=desired.device)

            return achieved_out, desired_out, step_out

    normalizer.append(NormalizerObj(obj_mean, obj_std))

    model.obj_traj = obj_traj.to('cuda')
    model.obj_traj.eval()

    for _ in range(1):
        state_all = dummy_env.reset()
        for i_step in range(config['episode_length']):

            model.to_cpu()

            obs = [K.tensor(obs, dtype=K.float32).unsqueeze(0) for obs in state_all['observation']]
            goal = K.tensor(state_all['desired_goal'], dtype=K.float32).unsqueeze(0)
            if i_step == 0:
                objtraj_goal = goal
            # Observation normalization
            obs_goal = []
            obs_goal.append(K.cat([obs[0], objtraj_goal], dim=-1))
            if normalizer[0] is not None:
                obs_goal[0] = normalizer[0].preprocess_with_update(obs_goal[0])

            action = model.select_action(obs_goal[0], noise).cpu().numpy().squeeze(0)
            
            action_to_env = np.zeros_like(dummy_env.action_space.sample())
            action_to_env[0:action.shape[0]] = action

            next_state_all, _, _, _ = dummy_env.step(action_to_env)

            # Move to the next state
            state_all = next_state_all

    #memory initilization  
    if her:
        sample_her_transitions = make_sample_her_transitions('future', 4, her_reward_fun)
    else:
        sample_her_transitions = make_sample_her_transitions('none', 4, her_reward_fun)

    buffer_shapes = {
        'o' : (config['episode_length'], dummy_env.observation_space.spaces['observation'].shape[1]*2),
        'ag' : (config['episode_length'], dummy_env.observation_space.spaces['achieved_goal'].shape[0]),
        'g' : (config['episode_length'], dummy_env.observation_space.spaces['desired_goal'].shape[0]),
        'u' : (config['episode_length']-1, action_space[2].shape[0])
        }
    memory = ReplayBuffer(buffer_shapes, MEM_SIZE, config['episode_length'], sample_her_transitions)

    experiment_args = ((envs, envs_test, envs_render), memory, noise, config, normalizer, None)

    print("0.20 - 0.25 - boundary_sample iff original_goal update norm")

    return model, experiment_args

def back_to_dict(state, config):

    goal_len = config['observation_space'].spaces['desired_goal'].shape[0]
    obs_len = config['observation_space'].spaces['observation'].shape[1]
    n_agents = config['observation_space'].spaces['observation'].shape[0]

    state_dict = {}
    state_dict['achieved_goal'] = state[:,0:goal_len]
    state_dict['desired_goal'] = state[:,goal_len:goal_len*2]
    state_dict['observation'] = state[:,goal_len*2::].reshape(-1,n_agents,obs_len).swapaxes(0,1)

    return state_dict

def rollout(env, model, noise, config, normalizer=None, render=False, step=(1,5), boundary_sample=False):
    trajectories = []
    for i_agent in range(2):
        trajectories.append([])
    
    # monitoring variables
    episode_reward = np.zeros(env.num_envs)
    frames = []
    
    state_all = env.reset()
    state_all = back_to_dict(state_all, config)

    step_l, step_h = step
    for i_step in range(config['episode_length']):

        model.to_cpu()

        obs = [K.tensor(obs, dtype=K.float32) for obs in state_all['observation']]
        goal = K.tensor(state_all['desired_goal'], dtype=K.float32)
        if i_step == 0:
            if (np.random.rand() < 0.20 and not boundary_sample) or (step_h >= config['episode_length']):
                objtraj_goal = goal
                original_goal = True
            else:
                original_goal = False
                achieved_goal = K.tensor(state_all['achieved_goal'], dtype=K.float32)

                objtraj_goal = []
                goal_len_per_obj = goal.shape[1]//model.n_objects
                for i_object in range(model.n_objects):
                    
                    achieved_goal_per_obj = achieved_goal[:,i_object*goal_len_per_obj:(i_object+1)*goal_len_per_obj]
                    goal_per_obj = goal[:,i_object*goal_len_per_obj:(i_object+1)*goal_len_per_obj]

                    normed_achieved_goal_per_obj, normed_goal_per_goal, normed_step = normalizer[2].process(achieved_goal_per_obj, 
                                                                                                            goal_per_obj, 
                                                                                                            K.randint(low=step_l,high=step_h+1,size=(achieved_goal_per_obj.shape[0],1)))
                    with K.no_grad():
                        objtraj_goal_per_obj = model.obj_traj(normed_achieved_goal_per_obj.to('cuda'), 
                                                              normed_goal_per_goal.to('cuda'), 
                                                              normed_step.to('cuda'))

                    objtraj_goal.append(objtraj_goal_per_obj.cpu())
                objtraj_goal = K.cat(objtraj_goal, dim=-1)
 
        # Observation normalization
        obs_goal = []
        obs_goal.append(K.cat([obs[0], objtraj_goal], dim=-1))
        if normalizer[0] is not None:
            #if not original_goal:
            #    obs_goal[0] = normalizer[0].preprocess(obs_goal[0])
            #else:
            obs_goal[0] = normalizer[0].preprocess_with_update(obs_goal[0])
        
        for i_agent in range(1, len(model.object_Qfunc)+1):
            obs_goal.append(K.cat([obs[1], objtraj_goal], dim=-1))
            # OR # obs_goal.append(K.cat([obs[1], goal], dim=-1))
            if normalizer[i_agent] is not None:
                obs_goal[i_agent] = normalizer[i_agent].preprocess(obs_goal[i_agent])

        action = model.select_action(obs_goal[0], noise).cpu().numpy()

        action_to_env = np.zeros((len(action), len(env.action_space.sample())))
        action_to_env[:,0:action.shape[1]] = action

        next_state_all, reward, done, info = env.step(action_to_env)
        next_state_all = back_to_dict(next_state_all, config)
        reward = K.tensor(reward, dtype=dtype).view(-1,1)

        next_obs = [K.tensor(next_obs, dtype=K.float32)for next_obs in next_state_all['observation']]
        
        # Observation normalization
        next_obs_goal = []
        next_obs_goal.append(K.cat([next_obs[0], objtraj_goal], dim=-1))
        if normalizer[0] is not None:
            next_obs_goal[0] = normalizer[0].preprocess(next_obs_goal[0])

        for i_agent in range(1, len(model.object_Qfunc)+1):
            next_obs_goal.append(K.cat([next_obs[1], objtraj_goal], dim=-1))
            if normalizer[i_agent] is not None:
                next_obs_goal[i_agent] = normalizer[i_agent].preprocess(next_obs_goal[i_agent])

        # for monitoring        
        episode_reward += reward.squeeze(1).cpu().numpy()
        
        r_intr = 0
        for i_agent in range(1, len(model.object_Qfunc)+1):
            r_intr += model.get_obj_reward(obs_goal[i_agent], next_obs_goal[i_agent], i_agent-1)

        episode_reward += (r_intr + reward).squeeze(1).cpu().numpy()

        for i_agent in range(2):
            state = {
                'observation'   : state_all['observation'][i_agent].copy(),
                'achieved_goal' : state_all['achieved_goal'].copy(),
                'desired_goal'  : objtraj_goal.numpy().copy()
                }
            
            trajectories[i_agent].append((state.copy(), action_to_env, reward))

        goal_a = state_all['achieved_goal']
        goal_b = objtraj_goal.numpy().copy()#state_all['desired_goal']
        ENV_NAME = config['env_id'] 
        if 'Rotate' in ENV_NAME:
            quat_a = goal_a[:,3:]
            quat_b = goal_b[:,3:]

        # Move to the next state
        state_all = next_state_all

        # Record frames
        if render:
            frames.append(env.render(mode='rgb_array')[0])

    if 'Rotate' in ENV_NAME:
        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        distance = (d_rot < 0.1).astype(np.float32)
    else:
        distance = np.linalg.norm(goal_a - goal_b, axis=-1)
        distance = (distance < 0.05).astype(np.float32)
    

    obs, ags, goals, acts = [], [], [], []

    for trajectory in trajectories:
        obs.append([])
        ags.append([])
        goals.append([])
        acts.append([])
        for i_step in range(config['episode_length']):
            obs[-1].append(trajectory[i_step][0]['observation'])
            ags[-1].append(trajectory[i_step][0]['achieved_goal'])
            goals[-1].append(trajectory[i_step][0]['desired_goal'])
            if (i_step < config['episode_length'] - 1): 
                acts[-1].append(trajectory[i_step][1])

    trajectories = {
        'o'    : np.concatenate(obs,axis=-1).swapaxes(0,1),
        'ag'   : np.asarray(ags)[0,].swapaxes(0,1),
        'g'    : np.asarray(goals)[0,].swapaxes(0,1),
        'u'    : np.asarray(acts)[0,].swapaxes(0,1),
    }

    info = np.asarray([i_info['is_success'] for i_info in info])

    return trajectories, episode_reward, info, distance

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    envs, memory, noise, config, normalizer, _ = experiment_args
    envs_train, envs_test, envs_render = envs
    if envs_test is None:
        envs_test = envs_train
    
    N_EPISODES = config['n_episodes'] if train else config['n_episodes_test']
    N_CYCLES = config['n_cycles']
    N_BATCHES = config['n_batches']
    N_TEST_ROLLOUTS = config['n_test_rollouts']
    BATCH_SIZE = config['batch_size']
    
    episode_reward_all = []
    episode_success_all = []
    episode_distance_all = []
    episode_reward_all_2 = []
    episode_success_all_2 = []
    episode_distance_all_2 = []
    episode_reward_mean = []
    episode_success_mean = []
    episode_distance_mean = []
    episode_reward_mean_2 = []
    episode_success_mean_2 = []
    episode_distance_mean_2 = []
    critic_losses = []
    actor_losses = []

    best_succeess = -1

    obj_traj_step = 1
        
    for i_episode in range(N_EPISODES):
        
        episode_time_start = time.time()        
        episode_reward_cycle_train = []
        episode_succeess_cycle_train = []
        episode_distance_cycle_train = []
        if train:
            for i_cycle in range(N_CYCLES):
                
                trajectories, episode_reward, success, distance = rollout(envs_train, model, noise, config, normalizer, render=False,
                                                                   step=(1, obj_traj_step))
                memory.store_episode(trajectories.copy())   

                episode_reward_cycle_train.extend(episode_reward)
                episode_succeess_cycle_train.extend(success)
                episode_distance_cycle_train.extend(distance)

                for i_batch in range(N_BATCHES):  
                    model.to_cuda()

                    batch = memory.sample(BATCH_SIZE)
                    critic_loss, actor_loss = model.update_parameters(batch, normalizer)
                    if i_batch == N_BATCHES - 1:
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)

                model.update_target()

                _, _, _, success = rollout(envs_train, model, False, config, normalizer=normalizer, render=False, 
                                                            step=(obj_traj_step, obj_traj_step),boundary_sample=True)

                if success.mean() >= 0.25 and obj_traj_step < config['episode_length']:
                    obj_traj_step += 1
                    print(obj_traj_step)

            # <-- end loop: i_cycle
        plot_durations(np.asarray(critic_losses), np.asarray(actor_losses))

        episode_reward_cycle = []
        episode_succeess_cycle = []
        episode_distance_cycle = []
        episode_reward_cycle_2 = []
        episode_succeess_cycle_2 = []
        episode_distance_cycle_2 = []
        rollout_per_env = N_TEST_ROLLOUTS // config['n_envs']
        for i_rollout in range(rollout_per_env):
            render = config['render'] == 2 and i_episode % config['render'] == 0
            _, episode_reward, success, distance = rollout(envs_train, model, False, config, normalizer=normalizer, render=render, 
                                                           step=(1,config['episode_length']))
                
            episode_reward_cycle.extend(episode_reward)
            episode_succeess_cycle.extend(success)
            episode_distance_cycle.extend(distance)

        for i_rollout in range(rollout_per_env):
            render = config['render'] == 2 and i_episode % config['render'] == 0
            _, episode_reward, success, distance = rollout(envs_test, model, False, config, normalizer=normalizer, render=render, 
                                                           step=(1,config['episode_length']))
                
            episode_reward_cycle_2.extend(episode_reward)
            episode_succeess_cycle_2.extend(success)
            episode_distance_cycle_2.extend(distance)

        render = (config['render'] == 1) and (i_episode % config['render'] == 0) and (envs_render is not None)
        if render:
            for i_rollout in range(10):
                _, _, _, _ = rollout(envs_render, model, False, config, normalizer=normalizer, render=render, 
                                     step=(1,config['episode_length']))
        # <-- end loop: i_rollout 
            
        ### MONITORIRNG ###
        episode_reward_all.append(episode_reward_cycle)
        episode_success_all.append(episode_succeess_cycle)
        episode_distance_all.append(episode_distance_cycle)

        episode_reward_all_2.append(episode_reward_cycle_2)
        episode_success_all_2.append(episode_succeess_cycle_2)
        episode_distance_all_2.append(episode_distance_cycle_2)

        episode_reward_mean.append(np.mean(episode_reward_cycle))
        episode_success_mean.append(np.mean(episode_succeess_cycle))
        episode_distance_mean.append(np.mean(episode_distance_cycle))

        episode_reward_mean_2.append(np.mean(episode_reward_cycle_2))
        episode_success_mean_2.append(np.mean(episode_succeess_cycle_2))
        episode_distance_mean_2.append(np.mean(episode_distance_cycle_2))

        plot_durations(np.asarray(episode_reward_mean), np.asarray(episode_success_mean))

        if best_succeess < np.mean(episode_succeess_cycle):
            bestmodel_critic = model.critics[0].state_dict()
            bestmodel_actor = model.actors[0].state_dict()
            bestmodel_normalizer = copy.deepcopy(normalizer)
            best_succeess = np.mean(episode_succeess_cycle)
        
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%1 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, N_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Running mean of total reward - training: {}'.format(np.mean(episode_reward_cycle_train)))
                print('  | Success rate - training: {}'.format(np.mean(episode_succeess_cycle_train)))
                print('  | Distance - training: {}'.format(np.mean(episode_distance_cycle_train)))
                print('  | Running mean of total reward: {}'.format(episode_reward_mean[-1]))
                print('  | Success rate: {}'.format(episode_success_mean[-1]))
                print('  | Distance to target {}'.format(episode_distance_mean[-1]))
                print('  | Success rate on normal: {}'.format(episode_success_mean_2[-1]))
                print('  | Distance to target on normal {}'.format(episode_distance_mean_2[-1]))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
            
    # <-- end loop: i_episode
    if train:
        print('Training completed')
    else:
        print('Test completed')

    return (episode_reward_all, episode_success_all, episode_success_all_2), (bestmodel_critic, bestmodel_actor, bestmodel_normalizer)

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