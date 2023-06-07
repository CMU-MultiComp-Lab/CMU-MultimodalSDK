#read_dataset_by_folder.py
#reads a dataset from a dataset folder

import mmsdk
from mmsdk import mmdatasdk
import argparse


parser = argparse.ArgumentParser(description='Reading dataset from a folder')
parser.add_argument('path', metavar='path', type=str, 
                    help='the folder path to read dataset from')
args = parser.parse_args()
dataset=mmdatasdk.mmdataset(args.path)



print ("List of the computational sequences")
print (dataset.computational_sequences.keys())

