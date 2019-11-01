#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                   dataset: omniglot                                    #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 16                                        #
#                                 dataset_and_sampler.py                                 #
#========================================================================================#

#packages
from __future__ import print_function

import os
import shutil
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import find_items, index_classes, load_img

#root
ROOT = '../datasets/new_omniglot'
raw_folder = 'raw'
processed_folder = 'data'

#dataset
class Omniglot(Dataset):
	'''
	this piece of code is fucking driving me crazy!
	'''
	def __init__(self, mode):
		super(Omniglot, self).__init__()
		with open(os.path.join(ROOT, mode + '.txt')) as f:
			self.classes = f.read().replace('/', os.sep).splitlines()
		self.all_items = find_items(ROOT, self.classes)
		self.idx_classes = index_classes(self.all_items)

		paths, self.labels = zip(*[self.get_path_label(pl) for pl in range(len(self))])

		self.x = map(load_img, paths, range(len(paths)))
		self.x = list(self.x)

	def __getitem__(self, index):
		return self.x[index], self.labels[index]

	def __len__(self):
		return len(self.all_items)

	def get_path_label(self, index):
		filename = self.all_items[index][0]
		rot = self.all_items[index][-1]
		img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
		target = self.idx_classes[self.all_items[index][1] + self.all_items[index][-1]]
		
		return img, target


#sampler
class OmniglotBatchSampler():
	'''
	torch batch sampler for few shot learning
	methods:
		__init__
		__iter__
		__len__
	'''
	def __init__(self, labels, num_batches, num_classes, num_samples):
		self.num_batches = num_batches
		self.num_classes = num_classes
		self.num_samples = num_samples

		labels = np.array(labels)
		
		self.class_class = []

		for i in range(max(labels) + 1):
			class_i = np.argwhere(labels == i).reshape(-1)
			class_i = torch.from_numpy(class_i)
			self.classes_class.append(class_i)

	def __iter__(self):
		for b in range(self.num_batches):
			batch = []
			classes = torch.randperm(len(self.classes_class))[:self.num_classes]
			
			for c in classes:
				the_class = self.class_class[c]
				samples_in_class = torch.randperm(len(the_class))[:self.num_samples]
				batch.append(the_class[samples_in_class])

			batch = torch.stack(batch).t().reshape(-1)
			
			yield batch

	def __len__(self):
		return self.num_batches
