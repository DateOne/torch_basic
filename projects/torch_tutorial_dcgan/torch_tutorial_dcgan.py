#=============================================================================#
#                           dcgan, torch tutorial                             #
#                           author: Yi                                        #
#                           2019, Oct 8                                       #
#=============================================================================#

#packages and settings
from __future__ import print_function

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from IPython.display import HTML

manualSeed = 999
print('random seed: ', manualSeed)
random.seed(humanSeed)
torch.manual_seed(manualSeed)

#inputs and settings
dataroot = "../../datasets/celeba"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
gf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#dataloader
dataset = dset.ImageFolder(
	root=dataroot, 
	transform=transforms.Compose([
		transforms.Resize(image_size),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

#pre-visualize
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('training images')
plt.imshow(np.tranpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, nromalize=True).cpu(). (1, 2, 0)))
if path.exists('saved') is False:
	os.mkdir('saved')
plt.savefig('saved/pre-visualize.png')

#weight initialization
def weights_init(m):
	classname = m.__class__.__name__
    if classname.find('Conv') != -1:
    	nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
    	nn.init.normal_(m.weight.data, 1.0, 0.02)
    	nn.init.constant_(m.bias.data, 0)

#g
class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh())
	def forward(self, input):
		return self.main(input)

netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
	netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)

#d
class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid())
	def forward(self, input):
		return self.main(input)

netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
	netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)

#computation graph
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_lable = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

#train
print('starting training..')

for epoch in range(num_epochs):
	for i, data in enumerate(dataloader, 0):
		netD.zero_grad()
		real_cpu = data[0].to(device)
		b_size = real_cpu.size(0)
		label = torch.full((b_size,), real_label, device=device)
		output = netD(real_cpu).view(-1)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()
		noise = torch.randn(b_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		output = netD(fake.detach()).view(-1)
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
        	print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        		% (epoch, num_epochs, i, len(dataloader),
        			errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        	with torch.no_grad():
        		fake = netG(fixed_noise).detach().cpu()
        	img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
		
		iters += 1

#results
plt.figure(figsize=(10,5))
plt.title("generator and discriminator loss during training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('saved/loss')

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

ani.save('saved/animation.mp4')

real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig('saved/real.png')

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
plt.savefig('saved/fake.png')