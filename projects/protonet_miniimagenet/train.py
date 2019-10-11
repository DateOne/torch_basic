#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 11                                        #
#                                       train.py                                         #
#========================================================================================#

#packages
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time

from dataset_and_sampler import MiniImagenet, FSLBatchSampler
from model import ProtoNet
from utils import pprint, set_device, ensure_path, Avenger, euclidean_distance

#main
if __name__ == '__main__':
	parser = argparse.ArgumentParser('protonet auguments, train')
	parser.add_argument(
		'-e', '--epochs',type=int,
		help='a fixed point of time',
		default=200)
	parser.add_argument(
		'-i', '--iterations', type=int,
		help='repetition of a process',
		default=100)
	parser.add_argument(
		'-lr', '--learning_rate', type=float,
		help='measure people learning speed',
		default=0.001)
	parser.add_argument(
		'-wtr', '--train_way', type=int,
		help='chuang wei, name of a restaurant',
		default=30)
	parser.add_argument(
		'-wte', '--test_way', type=int,
		help='brother of chuang wei',
		default=5)
	parser.add_argument(
		'-s', '--shot', type=int,
		help='an action of shooting',
		default=1)
	parser.add_argument(
		'-q', '--query', type=int,
		help='a question in mind',
		default=15)
	parser.add_argument(
		'-d', '--device',
		help='a scheme to deceive',
		default='0')
	parser.add_argument(
		'-sv_r', '--save_root',
		help='to protect the tree',
		default='save')
	args = parser.parse_args()
	pprint(vars(args))

	set_device(args.device)
	ensure_path(args.save_root)

	training_dataset = MiniImagenet('train')
	training_sampler = FSLBatchSampler(
		training_dataset.labels,
		num_batches=args.iterations,
		num_classes=args.train_way,
		num_samples=args.shot + args.query)
	training_dataloader = DataLoader(
		dataset=training_dataset,
		batch_sampler=training_sampler,
		num_workers=8,
		pin_memory=True)

	validation_dataset = MiniImagenet('val')
	validation_sampler = FSLBatchSampler(
		validation_dataset.labels,
		num_batches=400,
		num_classes=args.test_way,
		num_samples=args.shot + args.query)
	validation_dataloader = DataLoader(
		dataset=validation_dataset,
		batch_sampler=validation_sampler,
		num_workers=8,
		pin_memory=True)

	model = ProtoNet().cuda()

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=20)

	def save_model(name):
		torch.save(model.state_dict(), os.path.join(args.save_root, name + '.pth'))

	tr_log = {}
	tr_log['args'] = vars(args)
	tr_log['training_loss'] = []
	tr_log['training_acc'] = []
	tr_log['validation_loss'] = []
	tr_log['validation_acc'] = []
	tr_log['best_acc'] = []

	since = time.time()

	for epoch in range(args.epochs):
		lr_scheduler.step()

		model.train()

		training_loss = Avenger()
		training_acc = Avenger()

		for i, batch in enumerate(training_dataloader):
			data, _ = [_.cuda() for _ in batch]   #this is very confusing
			p = args.shot * args.train_way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.train_way, -1).mean(dim=0)

			label = torch.arange(args.train_way).repeat(args.query)   #this is confusing
			label = label.type(torch.cuda.LongTensor)

			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()   #this is confusing

			print('=== epoch: {}, train: {}/{}, loss={:.4f} acc={:.4f} ==='.format(epoch, i, len(train_loader), loss.item(), acc))

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

		for i, batch in enumerate(validation_loader, 1):
			data, _ = [_.cuda() for _ in batch]
			p = args.shot * args.test_way
			data_shot, data_query = data[:p], data[p:]

			protos = model(data_shot)
			protos = protos.reshape(args.shot, args.test_way, -1).mean(dim=0)

			label = torch.arange(args.test_way).repeat(args.query)
			label = label.type(torch.cuda.LongTensor)

			logits = euclidean_distance(model(data_query), protos)
			loss = F.cross_entropy(logits, label)
			pred = torch.argmax(logits, dim=1)
			acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

			validation_loss.add(loss.item())
			validation_acc.add(acc)

		validation_loss = validation_loss.item()
		validation_acc = validation_acc.item()

		print('=== epoch {}, val, loss={:.4f} acc={:.4f} ===\n\n'.format(epoch, vl, va))

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
		print('\n\n===========================\ntraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))