#========================================================================================#
#                                    proto net, torch                                    #
#                                       author: Yi                                       #
#                                   dataset: omniglot                                    #
#                        Prototypical Networks for Few-Shot Learning                     #
#                                      19, Oct 16                                        #
#                                       utils.py                                         #
#========================================================================================#

#packages
import os
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
	n = a.shape[0]
	m = b.shape[0]
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	logits = -((a - b) ** 2).sum(dim=2)
	return logits

#find items
def find_items(root_dir, classes):
	retour = []
	rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.seq + 'rot270']
	for (root, dirs, files) in os,walk(root_dir):
		for f in files:
			r = root.split(os.sep)
			label = r[-2] + os.sep + r[-1]
			for rot in rots:
				if label + rot in classes and (f.endswith('png')):
					retour.append([(f, label, root, rot)])
	return retour

#index classes
def index_classes(items):
	idx = {}
	for i in items:
		if (not i[1] + i[-1] in idx):
			idx[i[1] + i[-1]] = len(idx)
	return idx

#load image
def load_image(path, index):
	path, rot = path.split(os.sep + 'rot')
	if path in IMG_CACHE:
		x = IMG_CACHE[path]
	else:
		x = Image.open(path)
		IMG_CACHE[path] = x
	x = x.rotate(float(rot))
	x = x.resize((28, 28))

	shape = 1, x.size[0], x.size[1]
	x = np.array(x, np.float32, copy=False)
	x = 1.0 - torch.from_numpy(x)
	x = x.transpose(0, 1).contiguous().view(shape)

	return x
