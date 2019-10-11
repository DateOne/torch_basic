#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 11                                        #
#                                 dataset_and_sampler.py                                 #
#========================================================================================#

#packages
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#root
ROOT = '../../datasets'

#dataset
class MiniImagenet(Dataset):
	'''
	torch miniimagenet dataset
	methods:
		__init__
		__getitem__
		__len__
	'''
	def __init__(self, mode):
		csv_path = os.path.join(ROOT, mode + '.csv')
		lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

		data_paths = []
		labels = []
		label_indicator = -1

		self.label_names = []

		for line in lines:
			path, label_name = line.split(',')
			path = os.path.join(ROOT, 'images', name)

			if label_name not in self.label_names:
				self.label_names.append(label_name)
				label_indicator += 1
			
			data_paths.append(path)
			labels.append(label_indicator)

		self.data_paths = data_paths
		self.labels = labels

		self.transform = transforms.Compose([
			transforms.Resize(84),
			transforms.CenterCrop(84),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])])

	def __getitem__(self, idx):
		path, label = self.data_paths[idx], self.labels[idx]
		image = self.transform(Image.open(path).convert('RGB'))
		return image, label

	def __len__(self):
		return len(self.labels)

#sampler
class FSLBatchSampler():
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
		self.classes_idx = []

		for i in range(max(labels) + 1):
			class_idcs = np.argwhere(labels == i).reshape(-1)
			class_idcs = torch.from_numpy(clas_idcs)
			self.classes_idcs.append(class_idcs)

	def __iter__(self):
		for one_batch in range(self.num_batches):
			batch = []
			classes = torch.randperm(len(self.classes_idcs))[:self.num_classes]
			for c in classes:
				samples_idcs = self.classes_idcs[c]
				idcs = torch.randperm(len(samples_idcs))[:self.num_samples]
				batch.append(sample_idcs[idcs])
			batch = torch.stack(batch).t().reshape(-1)
			yield batch

	def __len__(self):
		return self.num_batches