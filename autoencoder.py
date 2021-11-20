import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

Data_dir_train = "Some address" 
Data_dir_test = "Some address"

train_ds = ImageFolder(Data_dir_train, transform = transforms.Compose([transforms.Resize(image_size), 
transforms.CentreCrop(image_size), 
transforms.ToTensor(), 
transforms.Normalize(*stats)]))

test_ds = ImageFolder(Data_dir_test, transform = transform.Compose([transforms.Resize(image_size), 
transforms.CentreCrop(image_size), 
transforms.ToTensor(), 
transforms.Normalize(*stats)]))

data_loader_train = torch.utils.data.DataLoader(dataset = train_ds, batch_size = 8, shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset = train_ds, batch_size = 8, shuffle = True)

dataiter = iter(data_loader_train)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))

dataiter = iter(data_loader_test)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))

class Encoder(nn.Module):

	def __init__(self, encoded_space_dim, fc2_input_dim):
		super().__init__()
		self.encoder_cnn = nn.Sequential(
			nn.Conv2d(3, 8, 3, stride = 2, padding = 1),
			nn.ReLU(True),
			nn.Conv2d(8, 16, 3, stride = 2, padding = 1),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			nn.Conv2d(16, 32,3 , stride = 2, padding = 0),
			nn.ReLU(True)
		)
		self.flatten = nn.Flatten(start_dim = 1)
		self.encoder_lin = nn.Sequential(
			nn.Linear(39*39*32, 256),
			nn.ReLU(True),
			nn.Linear(256, encoded_space_dim)
		)
	def forward(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)
		return x

class Decoder(nn.Module):

	def __init__(self, encoded_space_dim, fc2_input_dim):
		super().__init()
		self.decoder_lin = nn.Sequential(
			nn.Linear(encoded_space_dim, 256),
			nn.ReLU(True),
			nn.Linear(256, 39*39*32),
			nn.ReLU(True)
		)
		self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (32, 39, 39))
		self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride = 2, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.convTranspose(16, 8 ,3, stride = 2, padding=1, output_padding=1), nn.BatchNorm2d(8), nn.ReLU(True), nn.ConvTranspose(8, 3, 3,stride = 2, padding=1, output_padding=1)
		)
	
	def forward(self, x):
		x = self.decoder_lin(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		x = torch.sigmoid(x)
		return x

loss_fn = torch.nn.MSELoss()
lr = 0.001
torch.manual_seed(0)

encoded_space_dim = 16
encoder = Encoder(encoded_space_dim=encoded_space_dim, fc2_input_dim = image_size)
decoder = Decoder(encoded_space_dim=encoded_space_dim, fc2_input_dim = image_size)

encoder.type

params_to_optimize = [ {'params': encoder.parameters()}, {'params': decoder.parameters()}]
optim = torch.optim.Adam(params_to_optimize, lr = lr, weight_decay = 1e-05)

encoder = to_device(encoder, device) 
decoder = to_device(decoder, device)

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
	encoder.train()
	decoder.train()
	train_loss = []
	for image_batch, _in dataloader:
		image_batch = image_batch.to(device)
		encoded_data = encoder(image_batch)
		decoded_data = decoder(encoded_data)
		loss = loss_fn(decoded_data, image_batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss.append(loss.detach().cpu().numpy())
	return np.mean(train_loss)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):
	encoder.eval()
	decoder.eval()
	with torch.no_grad():
		conc_out = []
		conc_label = []
		for image_batch, _ in dataloader:
			image_batch = image_batch.to(device)
			encoded_data = encoder(image_batch)
			decoded_data = decoder(encoded_data)
			conc_out.append(decoded_data.cpu())
			conc_label.append(image_batch,cpu())
		conc_out = torch.cat(conc_out)
		conc_label = torch.cat(conc_label)
		val_loss = loss_fn(conc_out, conc_labels)
	return val_loss.data
