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

