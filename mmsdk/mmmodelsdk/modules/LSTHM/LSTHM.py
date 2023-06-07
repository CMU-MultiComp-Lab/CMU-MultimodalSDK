import torch
import time
from torch import nn
import torch.nn.functional as F


class LSTHM(nn.Module):

	def __init__(self,cell_size,in_size,hybrid_in_size):
		super(LSTHM, self).__init__()
		self.cell_size=cell_size
		self.in_size=in_size
		self.W=nn.Linear(in_size,4*self.cell_size)
		self.U=nn.Linear(cell_size,4*self.cell_size)
		self.V=nn.Linear(hybrid_in_size,4*self.cell_size)

	def step(self,x,ctm1,htm1,ztm1):
		input_affine=self.W(x)
		output_affine=self.U(htm1)
		hybrid_affine=self.V(ztm1)
		
		sums=input_affine+output_affine+hybrid_affine

		#biases are already part of W and U and V
		f_t=F.sigmoid(sums[:,:self.cell_size])
		i_t=F.sigmoid(sums[:,self.cell_size:2*self.cell_size])
		o_t=F.sigmoid(sums[:,2*self.cell_size:3*self.cell_size])
		ch_t=F.tanh(sums[:,3*self.cell_size:])
		c_t=f_t*ctm1+i_t*ch_t
		h_t=F.tanh(c_t)*o_t
		return c_t,h_t

	
