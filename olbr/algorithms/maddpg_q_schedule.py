import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from olbr.agents.basic import BackwardDyn, RandomNetDist

import gym_wmgds as gym

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_BD(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, 
                 reward_fun=None, clip_Q_neg=None, nb_critics=2,
                 dtype=K.float32, device="cuda"):

        super(DDPG_BD, self).__init__()

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.out_func = out_func
        self.discrete = discrete
        self.regularization = regularization
        self.normalized_rewards = normalized_rewards
        self.dtype = dtype
        self.device = device
        #if isinstance(observation_space, (list, tuple)):
        #    observation_space = observation_space[0]
        self.observation_space = observation_space
        self.action_space = action_space
        self.clip_Q_neg = clip_Q_neg if clip_Q_neg is not None else -1./(1.-self.gamma)
        self.nb_critics = nb_critics
        self.actor_action_space = gym.spaces.Box(action_space[0].low[0], 
                                                 action_space[0].high[0], 
                                                 shape=(action_space[0].shape[0]//2,), dtype='float32')

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        self.actors.append(Actor(observation_space[1], self.actor_action_space, discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space[1], self.actor_action_space, discrete, out_func).to(device))
        self.actors_optim.append(optimizer(self.actors[0].parameters(), lr = actor_lr))

        self.actors.append(Actor(observation_space[1], self.actor_action_space, discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space[1], self.actor_action_space, discrete, out_func).to(device))
        self.actors_optim.append(optimizer(self.actors[1].parameters(), lr = actor_lr))

        hard_update(self.actors_target[0], self.actors[0])
        hard_update(self.actors_target[1], self.actors[1])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        for i_critic in range(self.nb_critics):
            self.critics.append(Critic(observation_space[0], action_space[0]).to(device))
            self.critics_target.append(Critic(observation_space[0], action_space[0]).to(device))
            self.critics_optim.append(optimizer(self.critics[i_critic].parameters(), lr = critic_lr))

        for i_critic in range(self.nb_critics):
            hard_update(self.critics_target[i_critic], self.critics[i_critic])

        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        print('seperaate Qs for multiQ')

    def to_cpu(self):
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cpu()
        self.device = 'cpu'

    def to_cuda(self):        
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cuda()
        self.device = 'cuda'    

    def select_action(self, state, exploration=False, goal_size=None):
        if goal_size is None:
            self.actors[0].eval()
            with K.no_grad():
                mu = self.actors[0](state.to(self.device))
            self.actors[0].train()
        else:
            self.actors[0].eval()
            self.actors[1].eval()
            goal = state[:,-goal_size::]
            obs = state[:,0:-goal_size]
            obs1 = K.cat([obs[:,0:obs.shape[1]//2], goal], dim=-1)
            obs2 = K.cat([obs[:,obs.shape[1]//2::], goal], dim=-1)
            with K.no_grad():
                mu1 = self.actors[0](obs1.to(self.device))
                mu2 = self.actors[1](obs2.to(self.device))
            mu = K.cat([mu1,mu2], dim=-1)
            self.actors[0].train()
            self.actors[1].train()            
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space[0].low[0]), int(self.action_space[0].high[0]))

        return mu

    def update_parameters(self, batch, normalizer=None):

        observation_space = self.observation_space[0] - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)
        
        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        if normalizer[0] is not None:
            s1 = normalizer[0].preprocess(s1)
            s1_ = normalizer[0].preprocess(s1_)


        s, s_, a = s1, s1_, a1
        o1 = K.cat([s[:, 0:(observation_space//2)], s[:, observation_space::]], dim=-1)
        o2 = K.cat([s[:, (observation_space//2):observation_space], s[:, observation_space::]], dim=-1)

        o = [o1, o2]

        o1_ = K.cat([s_[:, 0:observation_space//2], s_[:, observation_space::]], dim=-1)
        o2_ = K.cat([s_[:, observation_space//2:observation_space], s_[:, observation_space::]], dim=-1)

        a1_ = self.actors_target[0](o1_)
        a2_ = self.actors_target[1](o2_)

        a_ = K.cat([a1_, a2_],dim=-1)

        r = K.tensor(batch['r'], dtype=self.dtype, device=self.device)

        for i_critic in range(self.nb_critics):
            Q = self.critics[i_critic](s, a)       
            V = self.critics_target[i_critic](s_, a_).detach()

            target_Q = (V * self.gamma) + r[:,i_critic:i_critic+1]
            target_Q = target_Q.clamp(self.clip_Q_neg, 0.)

            loss_critic = self.loss_func(Q, target_Q)

            self.critics_optim[i_critic].zero_grad()
            loss_critic.backward()
            self.critics_optim[i_critic].step()

        # actor update
        for i_actor in range(2):
            a1 = self.actors[0](o[0])
            a2 = self.actors[1](o[1])

            a = K.cat([a1, a2],dim=-1)

            loss_actor = 0.
            for i_critic in range(self.nb_critics):
                loss_actor += - self.critics[i_critic](s, a).mean()
            
            if self.regularization:
                loss_actor += (self.actors[i_actor](o[i_actor])**2).mean()*1

            self.actors_optim[i_actor].zero_grad()        
            loss_actor.backward()
            self.actors_optim[i_actor].step()
                
        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        soft_update(self.actors_target[1], self.actors[1], self.tau)
        for i_critic in range(self.nb_critics):
            soft_update(self.critics_target[i_critic], self.critics[i_critic], self.tau)



