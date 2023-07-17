import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

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

        td_error = torch.abs(Q - Q_target).detach()
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
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset(seed=seed + ep)

        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def train(env_name, model, buffer, timesteps=200_000, batch_size=128,
          eps_max=0.1, eps_min=0.0, test_every=5000, seed=0):
    print(f"Training on: {env_name}, Device: {device()}, Seed: {seed}")

    env = gym.make(env_name)

    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0

    episodes = 0
    best_reward = -np.inf

    done = False
    state, _ = env.reset(seed=seed)

    for step in range(1, timesteps + 1):
        if done:
            done = False
            state, _ = env.reset(seed=seed)
            episodes += 1

        eps = eps_max - (eps_max - eps_min) * step / timesteps

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = model.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add((state, action, reward, next_state, int(done)))

        state = next_state

        if step > batch_size:
            if isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(batch_size)
                loss, td_error = model.update(batch)
            elif isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(batch_size)
                loss, td_error = model.update(batch, weights=weights)

                buffer.update_priorities(tree_idxs, td_error.numpy())
            else:
                raise RuntimeError("Unknown buffer")

            total_loss += loss
            loss_count += 1

            if step % test_every == 0:
                mean, std = evaluate_policy(env_name, model, episodes=10, seed=seed)

                print(f"Episode: {episodes}, Step: {step}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                if mean > best_reward:
                    best_reward = mean
                    model.save()

                rewards_total.append(mean)
                stds_total.append(std)

    return np.array(rewards_total), np.array(stds_total)


def run_experiment(config, use_priority=False, n_seeds=10):
    torch.manual_seed(0)
    mean_rewards = []

    for seed in range(n_seeds):
        if use_priority:
            buffer = PrioritizedReplayBuffer(**config["buffer"])
        else:
            buffer = ReplayBuffer(**config["buffer"])
        model = DQN(**config["model"])

        seed_reward, seed_std = train(seed=seed, model=model, buffer=buffer, **config["train"])
        mean_rewards.append(seed_reward)

    mean_rewards = np.array(mean_rewards)

    return mean_rewards.mean(axis=0), mean_rewards.std(axis=0)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='DQN training with PER on CartPole-v0 or LunarLander-v2',
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('env_name', metavar='env_name', type=str, help='name of the environment for training')
    parser.add_argument('--seeds', dest='seeds', default=10, help='number of seeds for training', type=int)
    args = parser.parse_args()

    if args.env_name == "CartPole-v0":
        config = {
            "buffer": {
                "state_size": 4,
                "action_size": 1,  # action is discrete
                "buffer_size": 50_000
            },
            "model": {
                "state_size": 4,
                "action_size": 2,
                "gamma": 0.99,
                "lr": 1e-4,
                "tau": 0.01
            },
            "train": {
                "env_name": "CartPole-v0",
                "timesteps": 50_000,
                "batch_size": 64,
                "test_every": 5000,
                "eps_max": 0.5,
                "eps_min": 0.05
            }
        }
    elif args.env_name == "LunarLander-v2":
        config = {
            "buffer": {
                "state_size": 8,
                "action_size": 1,  # action is discrete
                "buffer_size": 100_000
            },
            "model": {
                "state_size": 8,
                "action_size": 4,
                "gamma": 0.99,
                "lr": 1e-3,
                "tau": 0.001
            },
            "train": {
                "env_name": "LunarLander-v2",
                "timesteps": 500_000,
                "start_train": 10_000,
                "batch_size": 128,
                "test_every": 5000,
                "eps_max": 0.5
            }
        }
    else:
        raise RuntimeError(f"Unknown env_name argument: {args.env_name}")

    priority_config = deepcopy(config)
    priority_config["buffer"].update({"alpha": 0.7, "beta": 0.4})

    mean_reward, std_reward = run_experiment(config, n_seeds=args.seeds)
    mean_priority_reward, std_priority_reward = run_experiment(priority_config, use_priority=True, n_seeds=args.seeds)

    steps = np.arange(mean_reward.shape[0]) * config["train"]["test_every"]

    plt.plot(steps, mean_reward, label="Uniform")
    plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)
    plt.plot(steps, mean_priority_reward, label="Prioritized")
    plt.fill_between(steps, mean_priority_reward - std_priority_reward, mean_priority_reward + std_priority_reward, alpha=0.4)

    plt.legend()
    plt.title(config["train"]["env_name"])
    plt.xlabel("Transitions")
    plt.ylabel("Reward")
    plt.savefig(f"{config['train']['env_name']}.jpg", dpi=200, bbox_inches='tight')