import numpy as np
from matplotlib import pyplot as plt

import os
import time
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

import utils
from utils import cal_loss
from models import Encoder, Decoder
from vae import VAE

import torchvision
from torchvision import transforms


def main(args):

	if args.data == 'MNIST':

		data_path = '/home/szchen/Datasets/'

		input_dim = 28 * 28

		transform = transforms.Compose([transforms.ToTensor()])
		mnist = torchvision.datasets.MNIST(data_path, download=False, transform=transform, train=True)
		dataloader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle=True)

	encoder = Encoder(input_dim=input_dim, args=args)
	decoder = Decoder(output_dim=input_dim, args=args)

	model = VAE(encoder=encoder, decoder=decoder, args=args).cuda()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	total_loss = []

	for epoch in tqdm(range(args.max_epoch)):
		epoch_loss = []

		for input_data, label in dataloader:
			input_data = Variable(input_data.view(-1, input_dim)).cuda()

			predict_, z_mean, z_log_var = model(input_data)

			optimizer.zero_grad()
			loss = cal_loss(predict_, input_data, z_mean, z_log_var, args)
			epoch_loss.append(loss.cpu().data)

			loss.backward()
			optimizer.step()

		total_loss.append(np.mean(epoch_loss))

		if args.save_fig != None and (epoch + 1) % args.save_fig == 0:
			test_image = model.inference(16)
			test_image = test_image.view(-1, 28, 28).detach().cpu().numpy()
			utils.save_image(test_image, 'Epoch:{}.png'.format(epoch))

	if args.save_paras:
		if not os.path.exists('./param'):
			os.mkdir('./param')
		torch.save(model.state_dict(), './param/parameters.pt')

	utils.draw_loss_curve(total_loss)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='simple')
	parser.add_argument('--max_epoch', type=int, default=15)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--latent_dim', type=int, default=2)
	parser.add_argument('--channels', type=int, default=256)
	parser.add_argument('--save_fig', type=int, default=1)
	parser.add_argument('--save_paras', type=bool, default=True)
	parser.add_argument('--loss', type=str, default='origin')
	parser.add_argument('--data', type=str, default='MNIST')

	args = parser.parse_args()

	main(args)





