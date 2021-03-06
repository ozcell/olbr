import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy 

from olbr.agents.basic import BackwardDyn

import pdb

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class PPO_BD(object):
    def __init__(self, observation_space, action_space, optimizer, Actor, Critic, 
                 clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef,
                 eps=None, max_grad_norm=None, use_clipped_value_loss=True,
                 out_func=K.sigmoid, discrete=True, agent_id=0, object_Qfunc=None, backward_dyn=None, 
                 object_policy=None, reward_fun=None, masked_with_r=False, dtype=K.float32, device="cuda",
                 ):

        super(PPO_BD, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        
        optimizer, lr = optimizer
        actor_lr, _ = lr

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.discrete = discrete
        self.agent_id = agent_id

        self.object_Qfunc = object_Qfunc
        self.object_policy = object_policy

        self.dtype = dtype
        self.device = device
        self.masked_with_r = masked_with_r
        self.gamma = 0.98
        self.tau = 0.05
        self.loss_func = F.mse_loss

        # model initialization
        self.entities = []
        
        # actors and critics
        self.actors = []
        self.critics = []
        self.optims = []

        self.actors.append(Actor(observation_space, action_space[agent_id], discrete, out_func).to(device))
        self.critics.append(Critic(observation_space, None).to(device))

        self.optims.append(optimizer(self.critics[0].parameters(), lr=actor_lr, eps=eps))
        self.optims[0].add_param_group({"params": self.actors[0].parameters()})
        
        self.entities.extend(self.actors)
        self.entities.extend(self.critics)
        self.entities.extend(self.optims) 

        # backward dynamics model
        if backward_dyn is not None:
            self.backward = backward_dyn
            self.backward_optim = optimizer(self.backward.parameters(), lr = actor_lr)
            self.entities.append(self.backward)
            self.entities.append(self.backward_optim)

        # Learnt Q function for object
        if self.object_Qfunc is not None:
            self.object_Qfunc_target = copy.deepcopy(self.object_Qfunc)
            self.object_Qfunc_optim = optimizer(self.object_Qfunc.parameters(), lr = actor_lr)
            self.entities.append(self.object_Qfunc)
            self.entities.append(self.object_Qfunc_target)
            self.entities.append(self.object_Qfunc_optim)

        # Learnt policy for object
        if self.object_policy is not None:
            self.object_policy_target = copy.deepcopy(self.object_policy)
            self.object_policy_optim = optimizer(self.object_policy.parameters(), lr = actor_lr)
            self.entities.append(self.object_policy)
            self.entities.append(self.object_policy_target)
            self.entities.append(self.object_policy_optim)

        if reward_fun is not None:
            self.get_obj_reward = reward_fun
        else:
            self.get_obj_reward = self.reward_fun
        
        print('clipped between -1 and 0, and masked with abs(r), and + r')

    def to_cpu(self):
        for entity in self.entities:
            if type(entity) != type(self.optims[0]):
                entity.cpu()
        self.device = 'cpu'

    def to_cuda(self):        
        for entity in self.entities:
            if type(entity) != type(self.optims[0]):
                entity.cuda()
        self.device = 'cuda'    

    def select_action(self, state, exploration=False):
        with K.no_grad():
            value = self.critics[0](state.to(self.device))
            action_dist = self.actors[0](state.to(self.device))

        if exploration:
            action = action_dist.sample()
        else:
            action = action_dist.mode()

        #action = action.clamp(int(self.action_space[self.agent_id].low[0]), int(self.action_space[self.agent_id].high[0]))

        return value, action, action_dist.log_probs(action)

    def get_obj_action(self, state, exploration=False):
        self.object_policy.eval()
        with K.no_grad():
            mu = self.object_policy(state.to(self.device))
        self.object_policy.train()
        if exploration:
            mu = K.tensor(exploration.get_noisy_action(mu.cpu().numpy()), dtype=self.dtype, device=self.device)
        mu = mu.clamp(int(self.action_space[1].low[0]), int(self.action_space[1].high[0]))

        return mu

    def evaluate_actions(self, state, action, get_preactivations=False):       
        value = self.critics[0](state.to(self.device))
        action_dist = self.actors[0](state.to(self.device))

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropies().mean()

        if get_preactivations:
            action_preactivations = self.actors[0].get_preactivations(state.to(self.device))
        else:
            action_preactivations = None

        return value, action_log_probs, dist_entropy, action_preactivations

    def reward_fun(self, state, next_state):
        with K.no_grad():
            action = self.backward(state.to(self.device), next_state.to(self.device))
            opt_action = self.object_policy(state.to(self.device))

            reward = self.object_Qfunc(state.to(self.device), action) - self.object_Qfunc(state.to(self.device), opt_action)
        return reward.clamp(min=-1.0, max=0.0)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch, mini_batch_size=None)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, action_preactivations  = self.evaluate_actions(obs_batch, actions_batch, False)

                ratio = K.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = K.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -K.min(surr1, surr2).mean()
                #action_loss += (action_preactivations**2).mean()*0.001

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * K.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optims[0].zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.critics[0].parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actors[0].parameters(), self.max_grad_norm)
                self.optims[0].step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_backward(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]
        
        s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a2 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, action_space:]

        s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if normalizer[1] is not None:
            s2 = normalizer[1].preprocess(s2)
            s2_ = normalizer[1].preprocess(s2_)

        a2_pred = self.backward(s2, s2_)

        loss_backward = self.loss_func(a2_pred, a2)

        self.backward_optim.zero_grad()
        loss_backward.backward()
        self.backward_optim.step()

        return loss_backward.item()

    def update_object_parameters(self, batch, normalizer=None):

        observation_space = self.observation_space - K.tensor(batch['g'], dtype=self.dtype, device=self.device).shape[1]
        action_space = self.action_space[0].shape[0]

        V = K.zeros((len(batch['o']), 1), dtype=self.dtype, device=self.device)
        
        s2 = K.cat([K.tensor(batch['o'], dtype=self.dtype, device=self.device)[:, observation_space:],
                    K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        a2 = K.tensor(batch['u'], dtype=self.dtype, device=self.device)[:, action_space:]

        s2_ = K.cat([K.tensor(batch['o_2'], dtype=self.dtype, device=self.device)[:, observation_space:],
                     K.tensor(batch['g'], dtype=self.dtype, device=self.device)], dim=-1)

        if normalizer[1] is not None:
            s2 = normalizer[1].preprocess(s2)
            s2_ = normalizer[1].preprocess(s2_)

        s, s_, a = s2, s2_, a2
        a_ = self.object_policy_target(s_)
    
        r = K.tensor(batch['r'], dtype=self.dtype, device=self.device).unsqueeze(1)

        Q = self.object_Qfunc(s, a)       
        V = self.object_Qfunc_target(s_, a_).detach()

        target_Q = (V * self.gamma) + r
        target_Q = target_Q.clamp(-1./(1.-self.gamma), 0.)

        loss_critic = self.loss_func(Q, target_Q)

        self.object_Qfunc_optim.zero_grad()
        loss_critic.backward()
        self.object_Qfunc_optim.step()

        a = self.object_policy(s)

        loss_actor = -self.object_Qfunc(s, a).mean()
        loss_actor += (self.object_policy(s)**2).mean()*1

        self.object_policy_optim.zero_grad()        
        loss_actor.backward()
        self.object_policy_optim.step()
                
        return loss_critic.item(), loss_actor.item()

    def update_object_target(self):

        soft_update(self.object_policy_target, self.object_policy, self.tau)
        soft_update(self.object_Qfunc_target, self.object_Qfunc, self.tau)
