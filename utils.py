import os
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt


def cal_loss(predict, groundtruth, z_mean, z_log_var, args):

	if args.loss == 'origin'
		BCE = nn.BCELoss(reduction='sum')

		KL_Loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
		BCE_Loss = BCE(predict, groundtruth)

		loss = (KL_Loss + BCE_Loss) / predict.shape[0]
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