from torch.distributions import Categorical, MultivariateNormal
import torch.nn as nn
import numpy as np
import torch
import copy

class BaseAgent:
    def __init__(
        self,
        actor,
        critic,
        device,
        state_dim,
        action_dim,
        discrete_actions=True,
    ):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions

    def prepare_state(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        state = state.to(self.device)
        if len(state.size()) < 2:
            state = state.unsqueeze(0)
        return state.float()

    def getAction(self, state, action=None, grad=False):
        state = self.prepare_state(state)
        if not grad:
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                action_values = self.actor(state)
                state_values = self.critic(state).squeeze(-1)
            self.actor.train()
            self.critic.train()
        else:
            action_values = self.actor(state)
            state_values = self.critic(state).squeeze(-1)

        if self.discrete_actions: 
            print(action_values)
            dist = Categorical(action_values)
        else:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_values, cov_mat)

        if action is None:
            action = dist.sample()
            return_tensors = False
        else:
            action = action.to(self.device)
            return_tensors = True

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        if return_tensors:
            return action, action_logprob, dist_entropy, state_values
        else:
            return (
                action.cpu().data.numpy(),
                action_logprob.cpu().data.numpy(),
                dist_entropy.cpu().data.numpy(),
                state_values.cpu().data.numpy(),
            )
        
    def test(self, env, max_steps=500, loop=3):
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state = env.reset()[0]
                score = 0
                for idx_step in range(max_steps):
                    action, _, _, _ = self.getAction(state)
                    state, reward, done, trunc, info = env.step(action)
                    score += reward[0]
                    if done[0] or trunc[0]:
                        break
                rewards.append(score)
        mean_fit = np.mean(rewards)
        return mean_fit
    
    @staticmethod
    def clone_module(module):
        clone = copy.deepcopy(module)
        clone.load_state_dict(module.state_dict())
        return clone

    def clone(self):
        return BaseAgent(
            actor=self.clone_module(self.actor),
            critic=self.clone_module(self.critic),
            device=self.device,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_actions=self.discrete_actions
        )