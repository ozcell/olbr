import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from olbr.agents.basic import BackwardDyn, RandomNetDist

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_BD(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=K.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, agent_id=0, object_Qfunc=[], object_inverse=[], 
                 object_policy=[], reward_fun=None, clip_Q_neg=None,
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
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = agent_id
        self.object_Qfunc = object_Qfunc
        self.object_policy = object_policy
        self.object_inverse = object_inverse
        self.clip_Q_neg = clip_Q_neg if clip_Q_neg is not None else -1./(1.-self.gamma)

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        self.actors.append(Actor(observation_space, action_space[agent_id], discrete, out_func).to(device))
        self.actors_target.append(Actor(observation_space, action_space[agent_id], discrete, out_func).to(device))
        self.actors_optim.append(optimizer(self.actors[0].parameters(), lr = actor_lr))

        hard_update(self.actors_target[0], self.actors[0])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []

        for i in range(len(object_Qfunc)+1):
            self.critics.append(Critic(observation_space, action_space[agent_id]).to(device))
            self.critics_target.append(Critic(observation_space, action_space[agent_id]).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))

            hard_update(self.critics_target[i], self.critics[i])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

        # Add object functions to entities
        for entity in [*self.object_Qfunc, *self.object_policy, *self.object_inverse]:
            entity.eval()
            self.entities.append(entity)

        if reward_fun is not None:
            self.get_obj_reward = reward_fun
        else:
            self.get_obj_reward = self.reward_fun

        print('seperaate Qs')

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

    def select_action(self, state, exploration=False):
        self.actors[0].eval()
        with K.no_grad():
            mu = self.actors[0](state.to(self.device))
        self.actors[0].train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space[self.agent_id].low[0]), int(self.action_space[self.agent_id].high[0]))

        return mu

    def update_parameters(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)
        
        s1 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a1 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, 0:action_space]

        s1_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, 0:observation_space],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)
        
        s, s_, a = [], [], a1
        if normalizer[0] is not None:
            s.append(normalizer[0].preprocess(s1))
            s_.append(normalizer[0].preprocess(s1_))
        else:
            s.append(s1)
            s_.append(s1_)

        for i in range(len(self.object_Qfunc)):
            if normalizer[i+1] is not None:
                s.append(normalizer[i+1].preprocess(s2))
                s_.append(normalizer[i+1].preprocess(s2_))
            else:
                s.append(s2)
                s_.append(s2_)

        r_all = []
        r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)
        r_all.append(r)
        for i in range(len(self.object_Qfunc)):
            r_all.append(self.get_obj_reward(s[i+1], s_[i+1], i))

        a_ = self.actors_target[0](s_[0])
        for i in range(0, len(self.object_Qfunc)+1):
            
            # first critic for main rewards
            Q = self.critics[i](s[0], a)       
            V = self.critics_target[i](s_[0], a_).detach()

            target_Q = (V * self.gamma) + r_all[i]
            target_Q = target_Q.clamp(self.clip_Q_neg, 0.)

            loss_critic = self.loss_func(Q, target_Q)

            self.critics_optim[i].zero_grad()
            loss_critic.backward()
            self.critics_optim[i].step()

        # actor update
        a = self.actors[0](s[0])

        loss_actor = 0
        for i in range(0, len(self.object_Qfunc)+1):
            loss_actor += -self.critics[i](s[0], a).mean()
        
        if self.regularization:
            loss_actor += (self.actors[0](s[0])**2).mean()*1

        self.actors_optim[0].zero_grad()        
        loss_actor.backward()
        self.actors_optim[0].step()
                
        return loss_critic.item(), loss_actor.item()

    def update_target(self):

        soft_update(self.actors_target[0], self.actors[0], self.tau)
        for i_critic in range(0, len(self.object_Qfunc)+1):
            soft_update(self.critics_target[i_critic], self.critics[i_critic], self.tau)

    def reward_fun(self, state, next_state, index=0):
        with K.no_grad():
            action = self.object_inverse[index](state.to(self.device), next_state.to(self.device))
            opt_action = self.object_policy[index](state.to(self.device))

            reward = self.object_Qfunc[index](state.to(self.device), action) - self.object_Qfunc[index](state.to(self.device), opt_action)
        return reward.clamp(min=-1.0, max=0.0)


