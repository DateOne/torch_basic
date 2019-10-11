#==============================================================================#
#                                protonet_torch                                #
#                                  author: yi                                  #
#                     Prototypical Networks for Few-Shot Learning              #
#                              dataset: omniglot                               #
#                                   19, Oct 9                                   #
#==============================================================================#

#packages
from __future__ import print_function

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import transforms

from tqdm import tqdm

#inputs and settings
parser = argparse.ArgumentParser(description='torch prototypical network on omniglot arguments')

parser.add_argument(
	'-e', '--epochs', type=int,
	help='number of epochs',
	default=100)
parser.add_argument(
	'-i', '--iterations', type=int,
	help='number of iterations',
	default=100)
parser.add_argument(
	'-lr', '--learning_rate', type=float,
	help='learning rate',
	default=0.001)

parser = parser.parse_args()

root = os.path.join('../../datasets', 'omniglot')
exp_root = 'exp'

lr_scheduler_step = 20
lr_scheduler_gamma = 0.5

manual_seed = 111
torch.manual_seed=manual_seed

num_classes_iter_tr = 60
num_support_samples_tr = 5
num_query_samples_tr = 5

num_classes_iter_val = 5
num_support_samples_val = 5
num_query_samples_val = 15

#dataloader
transform=transforms.ToTensor()

background_dataset = torchvision.datasets.Omniglot(root='../../datasets/torch_omniglot', background=True, transform=transform, download=True)
evaluation_dataset = torchvision.datasets.Omniglot(root='../../datasets/torch_omniglot', background=False, transform=transform, download=True)

background_dset_parameters = {'num_samples': 19280, 'num_classes': 964, 'num_sample_per_class': 20}
evaluation_dset_parameters = {'num_samples': 13180, 'num_classes': 659, 'num_sample_per_class': 20}

class FSLBatchSampler(object):
	'''
	few shot learning sampler
	sample a batch of indices for few shot learning task
	'''
	def __init__(self, data_source, num_classes, num_samples, iterations):
		self.data_source = data_source
		self.num_classes = num_classes
		self.num_samples = num_samples
		self.iterations = iterations
		if self.data_source == background_dataset:
			self.dset_parameters = background_dset_parameters
		else:
			self.dset_parameters = evaluation_dset_parameters
	def __iter__(self):
		batch = []
		class_idcs = torch.randperm(self.dset_parameters['num_classes'])[0 : self.num_classes]
		for class_idx in class_idcs:
			sample_idcs = torch.randperm(self.dset_parameters['num_sample_per_class'])[0 : self.num_samples]
			for sample_idx in sample_idcs:
				batch.append(self.dset_parameters['num_sample_per_class'] * class_idx + sample_idx)
		return iter(batch)
	def __len__(self):
		return self.iterations

bg_sampler = FSLBatchSampler(
	background_dataset,
	num_classes_iter_tr,
	num_support_samples_tr + num_query_samples_tr,
	parser.iterations)
eval_sampler = FSLBatchSampler(
	evaluation_dataset,
	num_classes_iter_val,
	num_support_samples_val + num_query_samples_val,
	parser.iterations)

bg_dataloader = torch.utils.data.DataLoader(background_dataset, batch_sampler=bg_sampler)
eval_dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_sampler=eval_sampler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#protonet
def conv_block(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2))

class ProtoNet(nn.Module):
	def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
		super(ProtoNet, self).__init__()
		self.encoder = nn.Sequential(
			conv_block(x_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, hid_dim),
			conv_block(hid_dim, z_dim))
	def forward(self, x):
		x = self.encoder(x)
		return x.view(x.size(0), -1)

def euclidean_dist(x, y):
	n = x.size(0)
	m = y.size(0)
	d = x.size(1)
	
	x = x.unsqueeze(1).expand(n, m, d)
	y = y.unsqueeze(0).expand(n, m, d)
	return torch.pow(x - y, 2).sum(2)

def proto_loss(inputs, targets, num_support):
	input_cpu = input.to('cpu')
	target_cpu = target.to('cpu')

	classes = torch.unique(target_cpu)
	num_classes = len(classes)

	num_query = target_cpu.eq(classes[0].item()).sum().item() - num_support

	def get_support_idcs(c):
		return target_cpu.eq(c).nonzero()[:num_support].squeeze(1)
	support_idcs = list(map(get_support_idcs, classes))
	prototypes = torch.stack([input_cpu[idx].mean(0) for idx in support_idcs])

	def get_query_idcs(c):
		return target_cpu.eq(c).nonzero()[num_support:]
	query_idcs = torch.stack(list(map(get_query_idcs, classes))).view(-1)
	query_samples = input_cpu[query_idcs]

	dists = euclidean_dist(query_samples, prototypes)

	log_p_y = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)

	target_inds = torch.arange(0, n_classes)
	target_inds = target_inds.view(n_classes, 1, 1)
	target_inds = target_inds.expand(n_classes, n_query, 1).long()

	loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
	_, y_hat = log_p_y.max(2)
	acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

	return loss_val, acc_val

class ProtoNetLoss(nn.Module):
	def __init__(self, num_support):
		super(ProtoNetLoss, self).__init__()
		self.num_support = num_support
	def forward(self, input, target):
		return proto_loss(input, target, self.n_support)

#train
def train(opt, tr_dataloader, model, criterion, optimizer, lr_scheduler):
	train_loss = []
	train_acc = []

	for epoch in range(opt.epochs):
		print('===== epoch: {} ====='.format(epoch))
		tr_iterator = iter(tr_dataloader)
		model.train()
		for batch in tr_iterator:
			optimizer.zero_grad()
			x, y = batch
			x, y = x.to(device), y.to(device)
			output = model(x)
			loss, acc = criterion(output, y)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			train_acc.append(acc.item())
		avg_loss = np.mean(train_loss[-opt.iterations:])
		avg_acc = np.mean(train_acc[-opt.iterations:])
		print('avg train loss: {}, avg train acc: {}'.format(avg_loss, avg_acc))
		lr_scheduler.step()

#test
def test(opt, test_dataloader, model, criterion):
	avg_acc = []
	for epoch in range(10):
		test_iterator = iter(test_dataloader)
		for batch in test_iterator:
			x, y = batch
			x, y = x.to(device), y.to(device)
			output = model(x)
			_, acc = criterion(output, y)
			avg_acc.append(acc.item)
	avg_acc = np.mean(avg_acc)
	print('test acc: {}'.format(avg_acc))


#main
if not os.path.exists(exp_root):
	os.makedirs(exp_root)

model = ProtoNet().to(device)
criterion = ProtoNetLoss(num_support_samples_tr)
optimizer = optim.Adam(params=model.parameters(), lr=parser.learning_rate)
lr_scheduler = optim.lr_scheduler.StepLR(
	optimizer=optimizer,
	gamma=lr_scheduler_gamma,
	step_size=lr_scheduler_step)

print('====================== start training ======================\n\n')
train(parser, bg_dataloader, model, criterion, optimizer, lr_scheduler)

print('\n\n\n')

print('====================== start testing ======================\n\n')
criterion = ProtoNetLoss(num_support_samples_val)
test(parser, eval_dataloader, model, criterion)
