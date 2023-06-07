#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multimodal Language Analysis with Recurrent Multistage Fusion, Paul Pu Liang, Ziyin Liu, Amir Zadeh, Louis-Philippe Morency - https://arxiv.org/abs/1808.03920 

#in_dimensions: the list of dimensionalities of each modality 

#cell_size: lstm cell size

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#steps: number of iterations for the recurrent fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce

class RecurrentFusion(nn.Module):

	def __init__(self,in_dimensions,cell_size):
		super(RecurrentFusion, self).__init__()
		self.in_dimensions=in_dimensions
		self.cell_size=cell_size
		self.model=nn.LSTM(sum(in_dimensions),cell_size)
	def __call__(self,in_modalities,steps=1):
		return self.fusion(in_modalities,steps)

	def fusion(self,in_modalities,steps=1):
		bs=in_modalities[0].shape[0]
		model_input=torch.cat(in_modalities,dim=1).view(1,bs,-1).repeat([steps,1,1])
		hidden,cell = (torch.zeros(1, bs, self.cell_size),torch.zeros(1, bs, self.cell_size))
		for i in range(steps):
			outputs,last_states=self.model(model_input,[hidden,cell])
		return outputs,last_states[0],last_states[1]
		
	def forward(self, x):
		print("Not yet implemented for nn.Sequential")
		exit(-1)

if __name__=="__main__":
	print("This is a module and hence cannot be called directly ...")
	print("A toy sample will now run ...")
	
	from torch.autograd import Variable
	import torch.nn.functional as F
	import numpy

	inputx=Variable(torch.Tensor(numpy.zeros([32,40])),requires_grad=True)
	inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
	inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
	modalities=[inputx,inputy,inputz]
	
	fmodel=RecurrentFusion([40,12,20],100)
	
	out=fmodel(modalities,steps=5)

	print("Output")
	print(out[0])
	print("Toy sample finished ...")


