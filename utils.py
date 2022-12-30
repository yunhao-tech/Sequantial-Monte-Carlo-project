import numpy as np
import random
from math import pi,sqrt, cos, sin

def fk(q, l):
			'''
			forward_kinematics
			q: array (2,) containing the two angles
			l: array (2,) containing the two limb lengths 
			'''
			x = l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1])
			y = l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1])
			return np.array([x, y])

def pd(q, x, xg, dq, k, l):
	'''
	pd control
	q : angle at time t array(2,)
	x : position at time t array(2,)
	xg : target position array(2,)
	dq : control at t-1 array(2,)
	k : hyperparameter array(2,)
	l : link length array(2,)
	'''
	a = l[0]*np.sin(q[0])
	b = l[0]*np.cos(q[0])
	c = l[1] * np.sin(q[0] + q[1])
	d = l[1] * np.cos(q[0] + q[1])
	# jac = np.array([[ -l[0] * np.sin(q[0]) - l[1] * np.sin(q[0] + q[1]), -l[1] * np.sin(q[0] + q[1])],
	# 								[ l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]), l[1] * np.cos(q[0] + q[1])]])
	jac = np.array([[ -a-c, -c],
									[ b+d, d]])
	u = - k[0] * np.matmul(jac , (x - xg)) - k[1] * dq
	return u

def randpt_in_circle(radius):
	r = radius * sqrt(random.random())
	theta = random.random() * 2 * pi
	x = r * cos(theta)
	y = r * sin(theta)
	return np.array([x,y])

def generate_motor_babbling_episode(l, n_timesteps):
		q_0 = np.random.uniform(low= -np.pi, high= np.pi, size=(2,1))
		u = np.random.randn(2, n_timesteps-1)
		q = u.cumsum(axis=1) + q_0
		q = np.concatenate((q_0, q), axis=1)
		x = fk(q, l=l)
		z = x + np.random.normal(0, scale=0.001, size=(2, n_timesteps))
		assert q.shape == x.shape
		return q, x, z

def generate_pd_control_episode(l, n_timesteps):
		q_0 = np.random.uniform(low= -np.pi, high= np.pi, size=2)
		x_0 = fk(q=q_0,l=l)
		x_g = randpt_in_circle(radius=0.1) + x_0
		q_t = q_0.copy()
		q = []
		x = []
		u_t = 0
		for i in range(n_timesteps):
			q.append(q_t)
			x_t = fk(q=q_t,l=l)
			x.append(x_t)
			u_t = pd(q_t, x=x_t, xg=x_g, dq=u_t, k=[1,1e-2], l=l)
			q_t += u_t

		x = np.asarray(x)
		z = x + np.random.normal(0, scale=0.001, size=x.shape)
		return q_0, x_g, x, z

def flip(x, ratio=0.5):
	'''
	binary mask flip by xor
	x: array of mask (n_features, n_masks)
	'''
	# flip by xor
	size = len(x)
	mask = np.zeros_like(x)
	mask[:(int)(ratio*size)] = 1
	# each mask is shuffled differently
	for i in range(x.shape[-1]):
		np.random.shuffle(mask[:,i])
	return x ^ mask

def resample(samples, weights):
	'''systematic resampling'''
	# weight normalization
	w = np.array(weights)
	assert w.sum() > 0, 'all weights are zero'
	w /= w.sum()
	w = w.cumsum()
	M = len(samples)
	ptrs = (random.random() + np.arange(M)) / M
	new_samples = []
	i = 0
	j = 0
	while i < M:
		if ptrs[i] < w[j]:
			new_samples.append(samples[j])
			i += 1
		else:
			j += 1
	return np.asarray(new_samples)