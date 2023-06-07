#create_toy_computational_sequence.py
#this example shows how to create two toy computational sequences and put them together in a dataset

import mmsdk
from mmsdk import mmdatasdk
import numpy


def random_init(compseq,feat_dim):
	for vid_key in vid_keys:
		num_entries=numpy.random.randint(low=5,high=100,size=1)
		compseq[vid_key]={}
		compseq[vid_key]["features"]=numpy.random.uniform(low=0,high=1,size=[num_entries,feat_dim])
		#let's assume each video is one minute, hence 60 seconds. 
		compseq[vid_key]["intervals"]=numpy.arange(start=0,stop=60+0.000001,step=60./((2*num_entries)-1)).reshape([num_entries,2])



if __name__=="__main__":
	vid_keys=["video1","video2","video3","video4","video5","Hello","World","UG3sfZKtCQI"]
	
	#let's assume compseq_1 is some modality with a random feature dimension
	compseq_1_data={}
	compseq_1_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_1_data,compseq_1_feature_dim)
	compseq_1=mmdatasdk.computational_sequence("my_compseq_1")
	compseq_1.setData(compseq_1_data,"my_compseq_1")
	#let's assume compseq_1 is some other  modality with a random feature dimension
	compseq_2_data={}
	compseq_2_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_2_data,compseq_2_feature_dim)
	compseq_2=mmdatasdk.computational_sequence("my_compseq_2")
	compseq_2.setData(compseq_2_data,"my_compseq_2")


	#NOTE: if you don't want to manually input the metdata, set it by creating a metdata key-value dictionary based on mmsdk/mmdatasdk/configurations/metadataconfigs.py
	compseq_1.deploy("compseq_1.csd")
	compseq_2.deploy("compseq_2.csd")

	#now creating a toy dataset from the toy compseqs
	mydataset_recipe={"compseq_1":"compseq_1.csd","compseq_2":"compseq_2.csd"}
	mydataset=mmdatasdk.mmdataset(mydataset_recipe)
	#let's also see if we can align to compseq_1
	mydataset.align("compseq_1")



