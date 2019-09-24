import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt


def cal_loss(predict, groundtruth, z_mean, z_log_var, args):
	# print(predict.shape, groundtruth.shape)
	KL_Loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))

	if args.loss == 'origin':
		A_Loss = F.binary_cross_entropy(predict.view(-1, 28*28), groundtruth.view(-1, 28*28), reduction='sum')

		

	if args.loss == 'MSE':
		addition_Loss = nn.MSELoss(reduction='sum')
		A_Loss = addition_Loss(predict, groundtruth)

	loss = (KL_Loss + A_Loss) / predict.shape[0]
	return loss


def draw_loss_curve(loss):
	x = np.arange(len(loss))
	y = loss

	plt.plot(x, y, label='loss', color='red', linewidth=2)
	plt.clf()

	if not os.path.exists('./figure'):
		os.mkdir('./figure')

	plt.savefig('./figure/loss_curve.png')
	plt.close('all')


def save_image(image, name):
	image_num = len(image)

	col = 4
	raw = int(math.ceil(image_num / col))
	plt.figure()
	for idx in range(image_num):
		plt.subplot(raw, col, idx+1)
		plt.imshow(image[idx])
		plt.axis('off')

	if not os.path.exists('./figure'):
		os.mkdir('./figure')

	plt.savefig('./figure/' + name)
	plt.clf()
	plt.close('all')