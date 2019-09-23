import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE(nn.Module):
	def __init__(self, encoder, decoder,args):
		super(VAE, self).__init__()
		self.args = args
		self.latent_dim = args.latent_dim
		self.encoder = encoder
		self.decoder = decoder

	def sample_latent_noise(self, mean, log_var):

		std = torch.exp(0.5 * log_var)
		e = torch.randn(mean.shape).cuda()

		return mean + e * std # reparameter trick

	def forward(self, x):
		mean, log_var = self.encoder(x)
		noise = self.sample_latent_noise(mean, log_var)
		output = self.decoder(noise)
		return output, mean, log_var

	def inference(self, number):
		normal_noise = Variable(torch.randn([number, self.latent_dim])).cuda()
		return self.decoder(normal_noise)