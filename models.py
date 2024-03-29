import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
	def __init__(self, input_dim, args):
		super(Encoder, self).__init__()

		self.args = args
		self.input_dim = input_dim
		self.latent_dim = args.latent_dim

		# self.Net = nn.Sequtial()
		if args.model == 'simple': # only have the simplest model for now
			self.layer  = nn.Linear(input_dim, args.channels)

			# calculate the mean and log var of the latent variables
		self.cal_mean = nn.Linear(args.channels, args.latent_dim)
		self.cal_log_var = nn.Linear(args.channels, args.latent_dim)


	def forward(self, x):
		if self.args.model == 'simple':
			x = F.relu(self.layer(x))

		mean = self.cal_mean(x)
		log_var = self.cal_log_var(x)

		return mean, log_var


class Decoder(nn.Module):
	def __init__(self, output_dim, args):
		super(Decoder, self).__init__()

		self.args = args
		self.latent_dim = args.latent_dim
		self.output_dim = output_dim

		if args.model == 'simple': # only have the simplest model for now
			self.layer = nn.Linear(args.latent_dim, args.channels)

		self.output_layer = nn.Linear(args.channels, output_dim)

	def forward(self, x):
		if self.args.model == 'simple':
			x = F.relu(self.layer(x))

		output = torch.sigmoid(self.output_layer(x))

		return output