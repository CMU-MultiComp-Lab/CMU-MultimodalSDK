#download_dataset.py
#downloads a standard dataset from multicomp servers
 
import mmsdk
from mmsdk import mmdatasdk
import argparse


parser = argparse.ArgumentParser(description='Downloads a dataset from web')

parser.add_argument('dataset',
                    metavar='dataset',
                    default='cmu_mosei',
                    choices=['cmu_mosei', 'cmu_mosi', 'pom'],
                    help='download a standard dataset (cmu_mosei,cmu_mosi,pom)')

args = parser.parse_args()
choice={"cmu_mosei":mmdatasdk.cmu_mosei.highlevel,"cmu_mosi":mmdatasdk.cmu_mosi.highlevel,"pom":mmdatasdk.pom.highlevel}
labels={"cmu_mosei":mmdatasdk.cmu_mosei.labels,"cmu_mosi":mmdatasdk.cmu_mosi.labels,"pom":mmdatasdk.pom.labels}

dataset=mmdatasdk.mmdataset(choice[args.dataset],'./downloaded_dataset')
dataset.add_computational_sequences(labels[args.dataset],'./downloaded_dataset')

print ("List of the computational sequences in the downloaded dataset")
print (dataset.computational_sequences.keys())

