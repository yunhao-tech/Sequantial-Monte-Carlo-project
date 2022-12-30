import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *

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
				q_0, x_g, x, z = generate_pd_control_episode(l, n_timesteps)
				self.test['q0'].append(q_0)
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
			"target_pos": torch.tensor(self.test['xg'][index]),
			"true_pos": torch.tensor(self.test['x'][index]),
			"noisy_pos": torch.tensor(self.test['z'][index]),
		}}
		return sample

	def __len__(self):
		return len(self.x_0)