#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph, Amir Zadeh, Paul Pu Liang, Jonathan Vanbriesen, Soujanya Poria, Edmund Tong, Erik Cambria, Minghai Chen, Louis-Philippe Morency - http://www.aclweb.org/anthology/P18-1208

#pattern_model: a nn.Sequential model which will be used as core of the models inside the DFG

#in_dimensions: input dimensions of each modality

#out_dimension: output dimension of the pattern models

#efficacy_model: the core of the efficacy model

#in_modalities: inputs from each modality, the same order as in_dimensions

import torch
import time
from torch import nn
import torch.nn.functional as F
import copy
from six.moves import reduce
from itertools import chain,combinations
from collections import OrderedDict

class DynamicFusionGraph(nn.Module):

	def __init__(self,pattern_model,in_dimensions,out_dimension,efficacy_model):
		super(DynamicFusionGraph, self).__init__()

		self.num_modalities=len(in_dimensions)
		self.in_dimensions=in_dimensions
		self.out_dimension=out_dimension

		#in this part we sort out number of connections, how they will be connected etc.
		self.powerset=list(chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities+1)))[1:]

		#initializing the models inside the DFG
		self.input_shapes={tuple([key]):value for key,value in zip(range(self.num_modalities),in_dimensions)}
		self.networks={}
		self.total_input_efficacies=0
		for key in self.powerset[self.num_modalities:]:
			#connections coming from the unimodal components
			unimodal_dims=0
			for modality in key:
				unimodal_dims+=in_dimensions[modality]
			multimodal_dims=((2**len(key)-2)-len(key))*out_dimension
			self.total_input_efficacies+=2**len(key)-2
			#for the network that outputs key component, what is the input dimension
			final_dims=unimodal_dims+multimodal_dims
			self.input_shapes[key]=final_dims
			pattern_copy=copy.deepcopy(pattern_model)
			final_model=nn.Sequential(*[nn.Linear(self.input_shapes[key],list(pattern_copy.children())[0].in_features),pattern_copy])
			self.networks[key]=final_model
		#finished construction weights, now onto the t_network which summarizes the graph
		self.total_input_efficacies+=2**self.num_modalities-1
		self.t_in_dimension=unimodal_dims+(2**self.num_modalities-(self.num_modalities)-1)*out_dimension
		pattern_copy=copy.deepcopy(pattern_model)
		self.t_network=nn.Sequential(*[nn.Linear(self.t_in_dimension,list(pattern_copy.children())[0].in_features),pattern_copy])
		self.efficacy_model=nn.Sequential(*[nn.Linear(sum(in_dimensions),list(efficacy_model.children())[0].in_features),efficacy_model,nn.Linear(list(efficacy_model.children())[-1].out_features,self.total_input_efficacies)])
	
	def __call__(self,in_modalities):
		return self.fusion(in_modalities)	

	def fusion(self,in_modalities):

		bs=in_modalities[0].shape[0]
		outputs={}
		for modality,index in zip(in_modalities,range(len(in_modalities))):
			outputs[tuple([index])]=modality
		
		efficacies=self.efficacy_model(torch.cat([x for x in in_modalities],dim=1))
		efficacy_index=0
		for key in self.powerset[self.num_modalities:]:
			small_power_set=list(chain.from_iterable(combinations(key, r) for r in range(len(key)+1)))[1:-1]
			this_input=torch.cat([outputs[x]*efficacies[:,efficacy_index+y].view(-1,1) for x,y in zip(small_power_set,range(len(small_power_set)))],dim=1)
			outputs[key]=self.networks[key](this_input)
			efficacy_index+=len(small_power_set)

		small_power_set.append(tuple(range(self.num_modalities)))
		t_input=torch.cat([outputs[x]*efficacies[:,efficacy_index+y].view(-1,1) for x,y in zip(small_power_set,range(len(small_power_set)))],dim=1)
		t_output=self.t_network(t_input)
		return t_output,outputs,efficacies

	def forward(self, x):
		print("Not yet implemented for nn.Sequential")
		exit(-1)

if __name__=="__main__":
	print("This is a module and hence cannot be called directly ...")
	print("A toy sample will now run ...")
	
	from torch.autograd import Variable
	import torch.nn.functional as F
	import numpy

	inputx=Variable(torch.Tensor(numpy.array(numpy.zeros([32,40]))),requires_grad=True)
	inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
	inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
	inputw=Variable(torch.Tensor(numpy.array(numpy.zeros([32,25]))),requires_grad=True)
	modalities=[inputx,inputy,inputz,inputw]
	
	#a simple linear function without any activations
	pattern_model=nn.Sequential(nn.Linear(100,20))
	efficacy_model=nn.Sequential(nn.Linear(100,20))
	fmodel=DynamicFusionGraph(pattern_model,[40,12,20,25],20,efficacy_model)
	
	out=fmodel(modalities)
	print("Output")
	print(out[0].shape,out[2].shape)
	print("Toy sample finished ...")

