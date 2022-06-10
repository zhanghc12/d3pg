import numpy as np
import torch
import gym
from knn_ue.sum_tree import SumSegmentTree, MinSegmentTree
import random


class OriginalReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).
        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done, action_tp1):
        """
        add a new transition to the buffer
        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, action_tp1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize


    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, actions_tp1 = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, action_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(np.array(done, copy=False))
            actions_tp1.append(np.array(action_tp1, copy=False))

        return (torch.FloatTensor(np.array(obses_t)).to(self.device),
                torch.FloatTensor(np.array(actions)).to(self.device),
                torch.FloatTensor(np.array(rewards)).to(self.device),
                torch.FloatTensor(np.array(obses_tp1)).to(self.device),
                torch.FloatTensor(np.array(dones)).to(self.device),
                torch.FloatTensor(np.array(actions_tp1)).to(self.device)
                )

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(OriginalReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.
        See Also ReplayBuffer.__init__
        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 2.0

    def add(self, obs_t, action, reward, obs_tp1, done, action_tp1):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, action_tp1)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0.5):
        """
        Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self._storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.forward_label = np.zeros((max_size, state_dim + 1))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.non_terminal_state = np.zeros((max_size, state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((max_size, action_dim))
        self.not_fake_done = np.zeros((max_size, 1))
        self.timestep = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def clear(self):
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.non_terminal_state = np.zeros((self.max_size, self.state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((self.max_size, self.action_dim))
        self.not_fake_done = np.zeros((self.max_size, 1))
        self.timestep = np.zeros((self.max_size, 1))

    def get_all_samples(self):
        if self.size < self.max_size:
            inputs = np.concatenate(
                [self.state[:self.ptr, :], self.action[:self.ptr, :]], axis=1)
            delta_state = self.next_state[:self.ptr,
                                          :] - self.state[:self.ptr, :]
            labels = np.concatenate(
                (self.reward[:self.ptr, :], delta_state), axis=-1)
        else:
            inputs = np.concatenate([self.state, self.action], axis=1)
            delta_state = self.next_state - self.state
            labels = np.concatenate((self.reward, delta_state), axis=-1)
        return inputs, labels

    def get_all_reverse_samples(self):  # sample question
        if self.size < self.max_size:
            inputs = np.concatenate(
                [self.next_state[:self.ptr, :], self.action[:self.ptr, :]], axis=1)
            delta_state = self.state[:self.ptr, :] - \
                self.next_state[:self.ptr, :]
            labels = np.concatenate(
                (self.reward[:self.ptr, :], delta_state), axis=-1)
        else:
            inputs = np.concatenate([self.next_state, self.action], axis=1)
            delta_state = self.state - self.next_state
            labels = np.concatenate((self.reward, delta_state), axis=-1)
        return inputs, labels

    def add(self, state, action, next_state,
            reward, done, fake_done, timestep=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.next_action[self.ptr - 1] = action
        self.not_fake_done[self.ptr] = 1. - fake_done
        self.timestep[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if done == 0:
            self.non_terminal_state[self.non_terminal_ptr] = next_state
            self.non_terminal_ptr = (self.non_terminal_ptr + 1) % self.max_size
            self.non_terminal_size = min(
                self.non_terminal_size + 1, self.max_size)

    def sample_states(self, batch_size):
        # to make sure not done
        ind = np.random.randint(0, self.size, size=batch_size)
        return torch.FloatTensor(self.non_terminal_state[ind]).to(self.device)

    def sample_by_index(self, ind, return_np=False):
        # ind = np.random.randint(0, self.size - 1, size=batch_size)

        if return_np:
            return self.state[ind], self.action[ind]
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
            )

    def sample_include_next_actions(self, batch_size):
        ind = np.random.randint(0, self.size - 1, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.not_fake_done[ind]).to(self.device)
        )

    def sample_v3(self, batch_size, include_next_action=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.forward_label[ind]).to(self.device)

        )


    def sample(self, batch_size, include_next_action=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        if include_next_action:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.next_action[ind]).to(self.device)
            )

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def sample_all_np(self):
        ind = np.arange(0, self.size)
        return (self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind])

    def sample_np(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind])



    def sample_include_timestep(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.timestep[ind]).to(self.device)

        )



class PerturbedReplayBuffer(object):
    def __init__(self, state_dim, action_dim, stds, target_threshold=0, max_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.forward_label = np.zeros((max_size, state_dim + 1))
        self.perturbed_next_state = np.zeros((max_size, state_dim))
        self.perturbed_reward = np.zeros((max_size, 1))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.non_terminal_state = np.zeros((max_size, state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((max_size, action_dim))
        self.not_fake_done = np.zeros((max_size, 1))
        self.timestep = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.target_threshold = target_threshold
        self.stds = stds
        self.stds_gpu = torch.unsqueeze(torch.FloatTensor(self.stds).to(self.device), dim=0)

    def clear(self):
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.non_terminal_state = np.zeros((self.max_size, self.state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((self.max_size, self.action_dim))
        self.not_fake_done = np.zeros((self.max_size, 1))
        self.timestep = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state,
            reward, done, fake_done, timestep=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        #self.perturbed_next_state[self.ptr] = next_state + self.target_threshold * np.random.normal(np.zeros_like(next_state), np.ones_like(next_state))
        #self.perturbed_reward[self.ptr] = reward + self.target_threshold * np.random.normal(np.zeros_like(reward), np.ones_like(reward))

        self.perturbed_next_state[self.ptr] = next_state + self.target_threshold * np.random.normal(np.zeros_like(self.stds), self.stds)
        self.perturbed_reward[self.ptr] = reward + self.target_threshold * np.random.normal(np.zeros_like(reward), np.ones_like(reward))


        self.next_action[self.ptr - 1] = action
        self.not_fake_done[self.ptr] = 1. - fake_done
        self.timestep[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if done == 0:
            self.non_terminal_state[self.non_terminal_ptr] = next_state
            self.non_terminal_ptr = (self.non_terminal_ptr + 1) % self.max_size
            self.non_terminal_size = min(
                self.non_terminal_size + 1, self.max_size)



    def sample(self, batch_size, include_next_action=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.perturbed_next_state[ind]).to(self.device),
            torch.FloatTensor(self.perturbed_reward[ind]).to(self.device),
        )



class ModeledReplayBuffer(object):
    def __init__(self, state_dim, action_dim, target_threshold=0, max_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.forward_label = np.zeros((max_size, state_dim + 1))
        self.perturbed_next_state = np.zeros((max_size, state_dim))
        self.perturbed_reward = np.zeros((max_size, 1))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.non_terminal_state = np.zeros((max_size, state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((max_size, action_dim))
        self.not_fake_done = np.zeros((max_size, 1))
        self.timestep = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.target_threshold = target_threshold

    def clear(self):
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.non_terminal_state = np.zeros((self.max_size, self.state_dim))
        self.non_terminal_ptr = 0
        self.non_terminal_size = 0

        self.next_action = np.zeros((self.max_size, self.action_dim))
        self.not_fake_done = np.zeros((self.max_size, 1))
        self.timestep = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state,
            reward, done, fake_done, perturbed_next_state, perturbed_reward, timestep=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        #self.perturbed_next_state[self.ptr] = next_state + self.target_threshold * np.random.normal(np.zeros_like(next_state), np.ones_like(next_state))
        #self.perturbed_reward[self.ptr] = reward + self.target_threshold * np.random.normal(np.zeros_like(reward), np.ones_like(reward))

        self.perturbed_next_state[self.ptr] = perturbed_next_state
        self.perturbed_reward[self.ptr] = perturbed_reward


        self.next_action[self.ptr - 1] = action
        self.not_fake_done[self.ptr] = 1. - fake_done
        self.timestep[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if done == 0:
            self.non_terminal_state[self.non_terminal_ptr] = next_state
            self.non_terminal_ptr = (self.non_terminal_ptr + 1) % self.max_size
            self.non_terminal_size = min(
                self.non_terminal_size + 1, self.max_size)



    def sample(self, batch_size, include_next_action=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.perturbed_next_state[ind]).to(self.device),
            torch.FloatTensor(self.perturbed_reward[ind]).to(self.device),
        )



class RunningMeanStd:
    def __init__(self, shape, device):
        self.device = device
        self.shape = shape
        self.count = 1e-3

        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)

    def update(self, x):
        batch_mean = torch.mean(x, 0)
        batch_var = torch.var(x, 0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * \
            self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class MeanStdNormalizer:
    def __init__(self, device='cpu'):
        self.rms = None
        self.max = 1
        self.epsilon = 1e-8
        self.device = device
        self.min = -1

    def __call__(self, x):
        if self.rms is None:
            self.rms = RunningMeanStd(
                shape=(1,) + x.shape[1:], device=self.device)
        self.rms.update(x)
        return torch.clamp((x - self.rms.mean) / torch.sqrt(self.rms.var +
                                                            self.epsilon), min=self.min, max=self.max)


def test_td(env, policy, onpolicy_buffer):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0
    episode_reward = 0
    episode_step = 0

    while not done:
        episode_step += 1
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 0
        fake_done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 1
        onpolicy_buffer.add(
            state,
            action,
            next_state,
            reward,
            done_bool,
            fake_done_bool)
        episode_reward += reward
        iter += 1
        state = next_state

        if iter > 100000:
            break
        if done:
            state, done = eval_env.reset(), False
            episode_reward = 0
            episode_step = 0
    for _ in range(50000):
        policy.train_value(onpolicy_buffer, batch_size=256)
    onpolicy_buffer.clear()


def test_mc(env, policy, onpolicy_buffer):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0
    episode_reward = 0
    episode_step = 0

    states = []
    actions = []
    rewards = []
    timesteps = []

    while True: #not done:
        episode_step += 1
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 0
        fake_done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 1

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        timesteps.append(episode_step)

        iter += 1
        state = next_state
        if iter > 100000:
            break
        if done:
            for i in reversed(range(len(rewards) - 1)):
                rewards[i] = 0.99 * rewards[i + 1] + rewards[i]

            for state, action, reward, timestep in zip(
                    states, actions, rewards, reversed(timesteps)):
                onpolicy_buffer.add(
                    state,
                    action,
                    next_state,
                    reward,
                    done_bool,
                    fake_done_bool,
                    timestep)

            state, done = eval_env.reset(), False
            states = []
            actions = []
            rewards = []
            timesteps = []
            episode_step = 0
    for _ in range(100000):
        policy.train_value_mc(onpolicy_buffer, batch_size=256)
    onpolicy_buffer.clear()


def test_mc_v2(env, policy, onpolicy_buffer):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0

    rewards = []

    episode_rewards = []
    while not done:
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        rewards.append(reward)
        iter += 1
        state = next_state
        if iter > 100000:
            break
        if done:
            for i in reversed(range(len(rewards) - 1)):
                rewards[i] = 0.99 * rewards[i + 1] + rewards[i]

            episode_rewards.append(rewards[0])
            state, done = eval_env.reset(), False
            episode_step = 0
            rewards = []
    return np.mean(episode_rewards)


def test_mc_v3(env, policy):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0

    rewards = []
    states = []
    actions = []

    final_states = []
    final_actions = []
    final_rewards = []
    n_mc_cutoff = 350
    while not done:
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)

        iter += 1
        state = next_state
        if iter > 100000:
            break
        if done:
            for i in reversed(range(len(rewards) - 1)):
                rewards[i] = 0.99 * rewards[i + 1] + rewards[i]
            final_rewards = np.concatenate((final_rewards, rewards[:n_mc_cutoff]))
            final_states = final_states + states[:n_mc_cutoff]
            final_actions = final_actions + actions[:n_mc_cutoff]
            state, done = eval_env.reset(), False

            rewards = []
            states = []
            actions = []
            # print('reset', iter)
    return final_rewards, np.array(final_states), np.array(final_actions)


def test_td_cpu(env, policy, onpolicy_buffer):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0
    episode_reward = 0
    episode_step = 0

    while not done:
        print(episode_step)
        episode_step += 1
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 0
        fake_done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 1
        onpolicy_buffer.add(
            state,
            action,
            next_state,
            reward,
            done_bool,
            fake_done_bool)
        episode_reward += reward
        iter += 1
        state = next_state

        if iter > 1000:
            break
        if done:
            state, done = eval_env.reset(), False
            episode_reward = 0
            episode_step = 0
    for _ in range(5):
        policy.train_value(onpolicy_buffer, batch_size=256)
    onpolicy_buffer.clear()


def test_mc_cpu(env, policy, onpolicy_buffer):
    eval_env = gym.make(env)
    state, done, iter = eval_env.reset(), False, 0
    episode_reward = 0
    episode_step = 0

    states = []
    actions = []
    rewards = []
    timesteps = []

    while not done:
        print(episode_step)
        episode_step += 1
        action = policy.select_action(np.array(state))
        next_state, reward, done, _ = eval_env.step(action)
        done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 0
        fake_done_bool = float(
            done) if episode_step < eval_env._max_episode_steps else 1

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        timesteps.append(episode_step + 1)

        iter += 1
        state = next_state
        if iter > 1000:
            break
        if done:
            for i in reversed(range(len(rewards) - 1)):
                rewards[i] = 0.99 * rewards[i + 1] + rewards[i]

            for state, action, reward, timestep in zip(
                    states, actions, rewards, timesteps):
                onpolicy_buffer.add(
                    state,
                    action,
                    next_state,
                    reward,
                    done_bool,
                    fake_done_bool,
                    timestep)

            state, done = eval_env.reset(), False
            states = []
            actions = []
            rewards = []
            timesteps = []
            episode_step = 0
    for _ in range(5):
        policy.train_value_mc(onpolicy_buffer, batch_size=256)
    onpolicy_buffer.clear()


def calc_gradient_penalty(x, y_pred):
    gradients = calc_gradients_input(x, y_pred)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty


def calc_gradients_input(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    return gradients