import gym
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from memory.utils import device, set_seed
from memory.buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQN:
    def __init__(self, state_size, action_size, gamma, tau, lr):
        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device())
        self.target_model = deepcopy(self.model).to(device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(device())
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch

        Q_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * Q_next
        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach() # TODO: check td error
        loss = torch.mean((Q - Q_target)**2 * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)

        return loss.item(), td_error

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(env_name, agent, episodes=5, seed=0):
    env = gym.make(env_name)
    set_seed(env, seed=seed)

    returns = []
    for _ in range(episodes):
        done, state, total_reward = False, env.reset(), 0

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def train(env_name, model, buffer, timesteps=200_000, start_train=1000, batch_size=128,
          eps_max=0.1, eps_min=0.0, test_every=5000, seed=0):
    print("Training on: ", device())

    env = gym.make(env_name)
    set_seed(env, seed=seed)

    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0

    episodes = 0
    best_reward = -np.inf

    done, state = False, env.reset()

    for step in range(1, timesteps + 1):
        if done:
            done, state = False, env.reset()
            episodes += 1

        eps = eps_max - (eps_max - eps_min) * step / timesteps

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = model.act(state)

        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))

        state = next_state

        if step > start_train:
            if isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(batch_size)
                loss, td_error = model.update(batch)
            elif isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(batch_size)
                loss, td_error = model.update(batch, weights=weights)

                buffer.update_priorities(tree_idxs, td_error.numpy())
                buffer.beta = buffer.beta - (buffer.beta - 1.0) * step / timesteps
            else:
                raise RuntimeError("Unknown buffer")

            total_loss += loss
            loss_count += 1

            if step % test_every == 0:
                mean, std = evaluate_policy(env_name, model, episodes=5, seed=seed)

                print(f"Episode: {episodes}, Step: {step + 1}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                if mean > best_reward:
                    best_reward = mean
                    model.save()

                rewards_total.append(mean)
                stds_total.append(std)

    return np.array(rewards_total), np.array(stds_total)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reward = []
    for i in range(10):
        buffer = ReplayBuffer(state_size=4, action_size=1, buffer_size=50_000)
        model = DQN(4, 2, gamma=0.99, lr=1e-4, tau=0.01)
        mean_rewards, _ = train("CartPole-v0", model, buffer, timesteps=50_000, start_train=5_000, batch_size=64,
                             test_every=5000, eps_max=0.2, seed=i)
        reward.append(mean_rewards)

    priority_reward = []
    for i in range(10):
        buffer = PrioritizedReplayBuffer(state_size=4, action_size=1, buffer_size=50_000, alpha=0.8, beta=0.3)
        model = DQN(4, 2, gamma=0.99, lr=1e-4, tau=0.01)
        mean_rewards, _ = train("CartPole-v0", model, buffer, timesteps=50_000, start_train=5_000, batch_size=64,
                                test_every=5000, eps_max=0.2, seed=i)

        priority_reward.append(mean_rewards)

    reward, priority_reward = np.array(reward), np.array(priority_reward)

    mean_reward, std_reward = reward.mean(axis=0), reward.std(axis=0)
    mean_priority_reward, std_priority_reward = priority_reward.mean(axis=0), priority_reward.std(axis=0)

    plt.plot(np.arange(reward.shape[1]) * 5000, mean_reward, label="Uniform")
    plt.fill_between(np.arange(reward.shape[1]) * 5000, mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)

    plt.plot(np.arange(reward.shape[1]) * 5000, mean_priority_reward, label="Prioritized")
    plt.fill_between(np.arange(reward.shape[1]) * 5000, mean_priority_reward - std_priority_reward, mean_priority_reward + std_priority_reward, alpha=0.4)
    plt.title("CartPole-v0")
    plt.xlabel("Transitions")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("cartpole.jpg", dpi=200, bbox_inches='tight')
    plt.show()

