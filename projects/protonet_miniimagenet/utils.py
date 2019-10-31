#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                  dataset: miniimagenet                                 #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 11                                        #
#                                       utils.py                                         #
#========================================================================================#

#packages
import os
import shutil
import pprint

#pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
	_utils_pp.pprint(x)

#set device
def set_device(x):
	os.environ['CUDA_VISIBLE_DEVICES'] = x
	print('using gpu: ', x)

#ensure path
def ensure_path(path):
	if os.path.exists(path):
		if input('{} exists, remove it? ([y]/n)'.format(path)) != 'n':
			shutil.rmtree(path)
			os.makedirs(path)
	else:
		os.makedirs(path)

#averager
class Avenger():
	'''
	avengers assemble!
	methods:
		__init__
		add
		item
	'''
	def __init__(self):
		self.n = 0
		self.res = 0

	def add(self, x):
		self.res = (self.res * self.n + x) / (self.n + 1)
		self.n += 1

	def item(self):
		return self.res

#euclidean distance
def euclidean_distance(a, b):
	n = a.shape[0]   #number of queries
	m = b.shape[0]   #training_way
	
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	
	logits = -((a - b) ** 2).sum(dim=2)   #number of queries times training way with each element being their distance
	
	return logits