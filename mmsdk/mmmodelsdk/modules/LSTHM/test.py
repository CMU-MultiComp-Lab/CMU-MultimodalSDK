import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from LSTHM import LSTHM
import numpy


#in=40, 3 modalities,4 attentions
full_in=numpy.array(numpy.zeros([32,40]))
inputx=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,12]))
inputy=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,12]))
inputc=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,20]))
inputz=Variable(torch.Tensor(full_in),requires_grad=True)

fmodel=LSTHM(12,40,20)
c,h=fmodel.step(inputx,inputc,inputy,inputz)
print(c.shape,h.shape)
#a=numpy.array([[1,2,3],[4,5,6]])
#b=Variable(torch.Tensor(a))
#c=b.repeat(1,2)
#print(c.shape)
#print(F.softmax(c,dim=1))
#print(c.view(2,2,3)[0,1,:])


