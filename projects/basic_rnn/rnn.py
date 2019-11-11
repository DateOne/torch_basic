#===============#
#   basic rnn   #
#===============#

import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch = 2
batch = 100
learning_rate = 0.01

sequence_length = 28
num_layers = 2

input_size = 28
hidden_size = 128
num_classes = 10

train_dataset = torchvision.datasets.MNIST(
	root='../../data/',
	train=True,
	transform=transforms.ToTensor(),
	download=True)
test_dataset = torchvision.datasets.MNIST(
	root='../../data/',
	train=False,
	transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=batch,
	shuffle=True)
test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,
	batch_size=batch,
	shuffle=False)

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, sequence_length, input_size).to(device)
		labels = labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.reshape(-1, sequence_length, input_size).to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print('acc: {} %'.format(100 * correct / total))