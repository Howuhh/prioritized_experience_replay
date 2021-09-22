import torch
import random
import numpy as np

from collections import deque
from memory.tree import SumTree
from memory.utils import device


class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=1e-2, alpha=0.1, beta=0.1, beta_scheduler=None):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps # init priority as eps
        self.beta_scheduler = beta_scheduler if beta_scheduler is not None else lambda x: x

        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def reset_scheduler(self):
        self.beta_scheduler.calls = 0

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition index with priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        cumsums = np.random.uniform(0, self.tree.total, size=batch_size)

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        for i, cumsum in enumerate(cumsums):
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta_scheduler(self.beta)
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        return batch


class NStepReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, n_step=3, gamma=0.99):
        # n-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma
        self.n_step = n_step

        # state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        state, action = self.n_step_buffer[0][:2]
        reward, next_state, done = self.get_n_step_return(self.n_step_buffer, self.gamma)

        self.state[self.count] = torch.as_tensor(state, dtype=torch.float)
        self.action[self.count] = torch.as_tensor(action, dtype=torch.float)
        self.reward[self.count] = torch.as_tensor(reward, dtype=torch.float)
        self.next_state[self.count] = torch.as_tensor(next_state, dtype=torch.float)
        self.done[self.count] = torch.as_tensor(done, dtype=torch.float)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get_n_step_return(self, n_step_buffer, gamma):
        n_reward, n_next_state, n_done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            reward, next_state, done = transition[-3:]

            n_reward = reward + gamma * n_reward * (1 - done)

            if done:
                n_next_state, n_done = next_state, done

        return n_reward, n_next_state, n_done

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)
        batch = (
            self.state[sample_idxs].to(device()),
            self.action[sample_idxs].to(device()),
            self.reward[sample_idxs].to(device()),
            self.next_state[sample_idxs].to(device()),
            self.done[sample_idxs].to(device())
        )
        return batch