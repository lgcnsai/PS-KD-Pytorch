import torch.nn as nn
from torch.nn import functional as F

import torch


class Custom_CrossEntropy_PSKD(nn.Module):
	def __init__(self):
		super(Custom_CrossEntropy_PSKD, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, output, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(output)
		loss = (- targets * log_probs).mean(0).sum()
		return loss        
        
         
        
        
        
        
        
        

        
