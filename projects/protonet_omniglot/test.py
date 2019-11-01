#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                    dataset: ominglot                                   #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 16                                        #
#                                        test.py                                         #
#========================================================================================#

#packages
import argparse

import torch
from torch.utils.data import DataLoader

from utils import pprint, set_device, Avenger, euclidean_distance
from dataset_and_sampler import Omniglot, FSLBatchSampler
from model import ProtoNet

#main
if __name__ == '__main__':
	parser = argparse.ArgumentParser('omniglot dataset, protonet, testing')
	
	parser.add_argument(
		'-i', '--batch', type=int,
		default=100)
	parser.add_argument(
		'-w', '--way', type=int,
		default=5)
	parser.add_argument(
		'-s', '--shot', type=int,
		default=5)
	parser.add_argument(
		'-q', '--testing_validation_query', type=int,
		default=15)
	parser.add_argument(
		'-d', '--device',
		default='0')
	args = parser.parse_args()
	pprint(vars(args))

	MODEL_ROOT = 'save/best.pth'
	set_device(args.device)

	dataset = Omniglot('test')
	sampler = FSLBatchSampler(
		dataset.labels,
		num_batches=args.batch,
		num_classes=args.way,
		num_samples=args.shot + args.testing_validation_query)
	dataloader = DataLoader(
		dataset,
		batch_sampler=sampler,
		num_workers=8,
		pin_memory=True)

	model = ProtoNet().cuda()
	model.load_state_dict(torch.load(MODEL_ROOT))

	model.eval()

	test_acc = Avenger()

	for i, batch in enumerate(dataloader):
		data, _ = [_.cuda() for _ in batch]
		p = args.way * args.shot
		data_shot, data_query = data[:p], data[p:]

		protos = model(data_shot)
		protos = protos.reshape(args.shot, args.way, -1).mean(dim=0)

		logits = euclidean_distance(model(data_query), protos)

		label = torch.arange(args.way).repeat(args.testing_validation_query).type(torch.cuda.LongTensor)
		
		#label = torch.arange(0, args.way).view(args.way, 1, 1).expand(args.way, args.query, 1).long()

		pred = torch.argmax(logits, dim=1)
		acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
		test_acc.add(acc)

		print('=== batch {}: {:.2f}({:.2f}) ==='.format(i, test_acc.item() * 100, acc * 100))