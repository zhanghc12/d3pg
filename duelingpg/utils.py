import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.non_terminal_state = np.zeros((max_size, state_dim))
		self.non_terminal_ptr = 0
		self.non_terminal_size = 0

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def get_all_samples(self):
		if self.size < self.max_size:
			inputs = np.concatenate([self.state[:self.ptr, :], self.action[:self.ptr, :]], axis=1)
			delta_state = self.next_state[:self.ptr, :] - self.state[:self.ptr, :]
			labels = np.concatenate((self.reward[:self.ptr, :], delta_state), axis=-1)
		else:
			inputs = np.concatenate([self.state, self.action], axis=1)
			delta_state = self.next_state - self.state
			labels = np.concatenate((self.reward, delta_state), axis=-1)
		return inputs, labels

	def get_all_reverse_samples(self): # sample question
		if self.size < self.max_size:
			inputs = np.concatenate([self.next_state[:self.ptr, :], self.action[:self.ptr, :]], axis=1)
			delta_state = self.state[:self.ptr, :] - self.next_state[:self.ptr, :]
			labels = np.concatenate((self.reward[:self.ptr, :], delta_state), axis=-1)
		else:
			inputs = np.concatenate([self.next_state, self.action], axis=1)
			delta_state = self.state - self.next_state
			labels = np.concatenate((self.reward, delta_state), axis=-1)
		return inputs, labels

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		if done == 0:
			self.non_terminal_state[self.non_terminal_ptr] = next_state
			self.non_terminal_ptr = (self.non_terminal_ptr + 1) % self.max_size
			self.non_terminal_size = min(self.non_terminal_size + 1, self.max_size)

	def sample_states(self, batch_size):
		# to make sure not done
		ind = np.random.randint(0, self.size, size=batch_size)
		return torch.FloatTensor(self.non_terminal_state[ind]).to(self.device)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
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
		M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / tot_count
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
			self.rms = RunningMeanStd(shape=(1,) + x.shape[1:], device=self.device)
		self.rms.update(x)
		return torch.clamp((x - self.rms.mean) / torch.sqrt(self.rms.var + self.epsilon), min=self.min, max=self.max)