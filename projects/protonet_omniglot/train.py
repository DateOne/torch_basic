#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                   dataset: omniglot                                    #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 16                                        #
#                                       train.py                                         #
#========================================================================================#

#packages
import os
import argparse

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time

from dataset_and_sampler import Omniglot, OmniglotBatchSampler
from model import ProtoNet
from utils import pprint, set_device, ensure_path, Avenger, euclidean_distance

#main
if __name__ == '__main__':
	parser = argparse.ArgumentParser('omniglot dataset, protonet, training')
	parser.add_argument(
		'-e', '--epochs',type=int,
		help='number of epochs',
		default=100)
	parser.add_argument(
		'-b', '--batch', type=int,
		help='number of batches',
		default=100)
	parser.add_argument(
		'-lr', '--learning_rate', type=float,
		help='learning rate',
		default=0.001)
	parser.add_argument(
		'-lr_g', '-learning_rate_gamma', type=float,
		help='learning rate gamma',
		default=0.5)
	parser.add_argument(
		'-lr_s', '--learning_rate_step', type=int,
		help='learning rate step size',
		default=20)
	parser.add_argument(
		'-tr_w', '--training_way', type=int,
		help='number of ways for meta-tasks in training',
		default=60)
	parser.add_argument(
		'-w', '--way', type=int,
		help='number of ways for meta-tasks',
		default=5)
	parser.add_argument(
		'-s', '--shot', type=int,
		help='number of shots for meta-tasks',
		default=5)
	parser.add_argument(
		'-q', '--query', type=int,
		help='number of query samples for meta-tasks',
		default=5)
	parser.add_argument(
		'-t_val_q', '--testing_validation_query', type=int,
		help='number of query samples for meta-tasks in validation and testing',
		default=15)
	parser.add_argument(
		'-d', '--device',
		help='device information',
		default='0')
	parser.add_argument(
		'-sv_r', '--save_root',
		help='save root information (not the dataset root)',
		default='save')
	
	args = parser.parse_args()
	pprint(vars(args))

	set_device(args.device)
	ensure_path(args.save_root)
	ensure_path(os.path.join(args.save_root, 'models'))
	ensure_path(os.path.join(args.save_root, 'grads'))

	training_dataset = Omniglot('train')
	training_sampler = OmniglotBatchSampler(
		training_dataset.labels,
		num_batches=args.batch,
		num_classes=args.training_way,
		num_samples=args.shot + args.query)
	training_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_sampler,
		num_workers=8,
		pin_memory=True)

	validation_dataset = Omniglot('val')
	validation_sampler = OmniglotBatchSampler(
		validation_dataset.labels,
		num_batches=args.batch,
		num_classes=args.way,
		num_samples=args.shot + args.testing_validation_query)
	validation_dataloader = DataLoader(
		dataset=validation_dataset,
		batch_sampler=validation_sampler,
		num_workers=8,
		pin_memory=True)

	model = ProtoNet().cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=20)

	def save_model(name):
		if name == 'best' or 'last':
			torch.save(model.state_dict(), os.path.join(args.save_root, name + '.pth'))
		else:
			torch.save(model.state_dict(), os.path.join(args.save_root, 'models', name + '.pth'))

	tr_log = {}
	tr_log['args'] = vars(args)
	tr_log['training_loss'] = []
	tr_log['training_acc'] = []
	tr_log['validation_loss'] = []
	tr_log['validation_acc'] = []
	tr_log['best_acc'] = 0

	since = time.time()

	for epoch in range(args.epochs):
		lr_scheduler.step()

		model.train()

		training_loss = Avenger()
		training_acc = Avenger()

		for i, batch in enumerate(training_dataloader):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.training_way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.training_way, -1).mean(dim=0)

			label = torch.arange(args.training_way).repeat(args.query).type(torch.cuda.LongTensor)

			#label = torch.arange(0, args.train_way).view(args.train_way, 1, 1).expand(args.train_way, args.query, 1).long()

			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			print('=== epoch: {}, train: {}/{}, loss={:.4f} acc={:.4f} ==='.format(epoch + 1, i + 1, len(training_dataloader), loss.item(), acc))

			training_loss.add(loss.item())
			training_acc.add(acc)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		training_loss = training_loss.item()
		training_acc = training_acc.item()

		model.eval()

		validation_loss = Avenger()
		validation_acc = Avenger()

		for i, batch in enumerate(validation_dataloader):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.way, -1).mean(dim=0)

			label = torch.arange(args.way).repeat(args.testing_validation_query).type(torch.cuda.LongTensor)
			
			#label = torch.arange(0, args.validation_way).view(args.validation_way, 1, 1).expand(args.validation_way, args.validation_query, 1).long()

			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			validation_loss.add(loss.item())
			validation_acc.add(acc)

		validation_loss = validation_loss.item()
		validation_acc = validation_acc.item()

		print('=== epoch {}, val, loss={:.4f} acc={:.4f} ==='.format(epoch, validation_loss, validation_acc))

		if validation_acc > tr_log['best_acc']:
			tr_log['best_acc'] = validation_acc
			save_model('best')

		tr_log['training_loss'].append(training_loss)
		tr_log['training_acc'].append(training_acc)
		tr_log['validation_loss'].append(validation_loss)
		tr_log['validation_acc'].append(validation_acc)

		torch.save(tr_log, os.path.join(args.save_root, 'tr_log'))

		save_model('last')

		if (epoch + 1) % 20 == 0:
			save_model('epoch-{}'.format(epoch))

		time_elapsed = time.time() - since
		print('trainging complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
