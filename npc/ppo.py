import copy
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal

from .base_agent import BaseAgent

class PPO:
    def __init__(
        self,
        agent,
        index,
        batch_size=64,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
    ):
        self.agent = agent
        self.actor = agent.actor
        self.critic = agent.critic
        
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.algo = "PPO" # Used for mutation
        self.mut = "None" # Most recent mutation to agent

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # Create the optimizer
        actor_params = set(agent.actor.parameters())
        critic_params = set(agent.critic.parameters())
        combined_params = actor_params.union(critic_params)
        self.optimizer = optim.Adam([
            {"params": list(combined_params), "lr": self.lr}
        ])
        self.optimizer_type = self.optimizer

    def getAction(self, state, action=None, grad=False):
        return self.agent.getAction(state, action, grad)

    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        print([arr for arr in experiences])
        #experiences = [torch.from_numpy(np.array(exp.cpu())) for exp in experiences]
        states, actions, log_probs, rewards, dones, values, next_state = experiences
        # TODO: clean up
        print(f'# states: {len(states)}, state shape: {states[0].shape}')
        states = torch.stack([state.squeeze() for state in states])
        print(f'stacked states shape: {states.shape}')
        actions = torch.from_numpy(np.array(actions))
        log_probs = torch.from_numpy(np.array(log_probs))
        rewards = torch.from_numpy(np.array(rewards))
        values = torch.from_numpy(np.array(values))
        dones = torch.from_numpy(np.array(dones)).long()

        # Bootstrapping
        with torch.no_grad():
            num_steps = rewards.size(0)
            next_state = self.agent.prepare_state(next_state)
            next_value = self.agent.critic(next_state).reshape(1, -1).cpu()
            advantages = torch.zeros_like(rewards)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lambda
                )
            returns = advantages + values

        states = states.reshape((-1,) + self.agent.state_dim)
        if self.agent.discrete_actions:
            actions = actions.reshape(-1)
        else:
            actions = actions.reshape((-1, self.agent.action_dim))
        log_probs = log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        values = values.reshape(-1)

        device = self.agent.device
        states, actions, log_probs, advantages, returns, values = (
            states.to(device),
            actions.to(device),
            log_probs.to(device),
            advantages.to(device),
            returns.to(device),
            values.to(device),
        )

        num_samples = returns.size(0)
        batch_idxs = np.arange(num_samples)

        clipfracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]

                _, log_prob, entropy, value = self.getAction(
                    state=states[minibatch_idxs],
                    action=actions[minibatch_idxs],
                    grad=True,
                )

                logratio = log_prob - log_probs[minibatch_idxs]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                minibatch_advs = advantages[minibatch_idxs]
                minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (
                    minibatch_advs.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -minibatch_advs * ratio
                pg_loss2 = -minibatch_advs * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.view(-1)
                v_loss_unclipped = (value - returns[minibatch_idxs]) ** 2
                v_clipped = values[minibatch_idxs] + torch.clamp(
                    value - values[minibatch_idxs], -self.clip_coef, self.clip_coef
                )
                v_loss_clipped = (v_clipped - returns[minibatch_idxs]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # actor loss backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self, env, max_steps=500, loop=3):
        mean_fit = self.agent.test(env, max_steps, loop)
        self.fitness.append(mean_fit)
        return mean_fit
    
    def clone(self, index=None, wrap=False):
        if index is None:
            index = self.index
        if wrap:
            raise NotImplementedError("Wrapping not implemented")

        agent_clone = self.agent.clone()

        clone = type(self)(
            agent=agent_clone,
            index=index,
            batch_size=self.batch_size,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            update_epochs=self.update_epochs
        )

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone