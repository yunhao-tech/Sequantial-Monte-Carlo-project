import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union
from utils import *

class TwoArmLinkTrainDataset(Dataset):
	# `set_seed=None` means no seed fixed.  
	def __init__(self, n_tasks, n_episode, 
				n_timesteps, set_seed: Union[int, None]=None):
		super().__init__()
		self.q = []
		self.x = []
		self.z = []
		if set_seed is not None:
			np.random.seed(set_seed)
		l_of_tasks = np.random.normal(loc=1, scale=0.3, size=(n_tasks,2))
		for i in tqdm(range(n_tasks)):
			l = l_of_tasks[i]
			for _ in range(n_episode):
				q_, x, z = generate_motor_babbling_episode(l, n_timesteps)
				self.q.extend(q_)
				self.x.extend(x)
				self.z.extend(z)
		self.q = np.stack(self.q, axis=1).reshape(-1,2)
		self.x = np.stack(self.x, axis=1).reshape(-1,2)
		self.z = np.stack(self.z, axis=1).reshape(-1,2)
		assert self.q.shape == self.x.shape
		print("Two Arm Link Dataset Generation Finished!")
		
	def __getitem__(self, index):
		q = self.q[index]
		x = self.x[index]
		z = self.z[index]
		sample = {
			"angle": torch.tensor(q),
			"true_pos": torch.tensor(x),
			"noisy_pos": torch.tensor(z),
		}
		return sample

	def __len__(self):
		return len(self.q)

class TwoArmLinkTestDataset(Dataset):
	def __init__(self, n_tasks=100, n_episode=1, n_timesteps=200) -> None:
		super().__init__()
		self.l = []
		self.adapt = {
			'q':[],
			'x':[],
			'z':[]}
		self.test = {
			'q0':[],
			'x0':[],
			'xg':[],
			'x':[],
			'z':[],
		}
		l_of_tasks = np.random.normal(loc=1, scale=0.3, size=(n_tasks,2))
		for i in tqdm(range(n_tasks)):
			l = l_of_tasks[i]
			self.l.append(l)
			for j in range(n_episode):
				# generate motor babbling
				q, x, z = generate_motor_babbling_episode(l, n_timesteps)
				self.adapt['q'].append(q)
				self.adapt['x'].append(x)
				self.adapt['z'].append(z)
				# generate pd control
				q_0, x_0, x_g, x, z = generate_pd_control_episode(l, n_timesteps)
				self.test['q0'].append(q_0)
				self.test['x0'].append(x_0)
				self.test['xg'].append(x_g)
				self.test['x'].append(x)
				self.test['z'].append(z)

	def __getitem__(self, index):
		sample = {
		'adapt':{
			'angle': torch.tensor(self.adapt['q'][index]),
			'true_pos': torch.tensor(self.adapt['x'][index]),
			'noisy_pos': torch.tensor(self.adapt['z'][index]),
		},
		'test': {
			"init_angle": torch.tensor(self.test['q0'][index]),
			"init_pos": torch.tensor(self.test['x0'][index]),
			"target_pos": torch.tensor(self.test['xg'][index]),
			"true_pos": torch.tensor(self.test['x'][index]),
			"noisy_pos": torch.tensor(self.test['z'][index]),
		}}
		return sample

	def __len__(self):
		return len(self.l)