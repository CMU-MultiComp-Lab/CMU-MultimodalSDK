import sys
import h5py
import os
import json
from tqdm import tqdm
from mmsdk.mmdatasdk import log
from mmsdk.mmdatasdk.configurations.metadataconfigs import *
from mmsdk.mmdatasdk.computational_sequence.integrity_check import *

def read_CSD(resource,destination=None):

	if (resource is None): raise log.error("No resource specified for computational sequence!",error=True)	
	if os.path.isfile(resource) is False:
		log.error("%s file not found, please check the path ..."%resource,error=True)	
	try:
		h5handle=h5py.File('%s'%resource,'r')
	except: 
		raise log.error("%s resource is not a valid hdf5 computational sequence format ..."%resource,error=True)
	log.success ("Computational sequence read from file %s ..."%resource)
	return h5handle,dict(h5handle[list(h5handle.keys())[0]]["data"]),metadata_to_dict(h5handle[list(h5handle.keys())[0]]["metadata"])


#writing CSD files to disk
def write_CSD(data,metadata,rootName,destination,compression,compression_opts,full_chunk_shape):

	log.status("Writing the <%s> computational sequence data to %s"%(rootName,destination))
	if compression is not None:
		log.advise("Compression with %s and opts -%d"%(compression,compression_opts))
	#opening the file
	writeh5Handle=h5py.File(destination,'w')
	#creating the root handle
	rootHandle=writeh5Handle.create_group(rootName)

	#writing the data
	dataHandle=rootHandle.create_group("data")
	pbar = log.progress_bar(total=len(data.keys()),unit=" Computational Sequence Entries",leave=False)
	for vid in data:
		vidHandle=dataHandle.create_group(vid)
		if compression is not None:
			vidHandle.create_dataset("features",data=data[vid]["features"],compression=compression,compression_opts=compression_opts)
			vidHandle.create_dataset("intervals",data=data[vid]["intervals"],compression=compression,compression_opts=compression_opts)
		else:
			vidHandle.create_dataset("features",data=data[vid]["features"])
			vidHandle.create_dataset("intervals",data=data[vid]["intervals"])
			
		pbar.update(1)
	pbar.close()
	log.success("<%s> computational sequence data successfully wrote to %s"%(rootName,destination))
	log.status("Writing the <%s> computational sequence metadata to %s"%(rootName,destination))
	#writing the metadata
	metadataHandle=rootHandle.create_group("metadata")
	for metadataKey in metadata.keys():
		metadataHandle.create_dataset(metadataKey,(1,),dtype=h5py.special_dtype(vlen=unicode) if sys.version_info.major is 2 else h5py.special_dtype(vlen=str))
		cast_operator=unicode if sys.version_info.major is 2 else str
		metadataHandle[metadataKey][0]=cast_operator(json.dumps(metadata[metadataKey]))
	writeh5Handle.close()

	log.success("<%s> computational sequence metadata successfully wrote to %s"%(rootName,destination))
	log.success("<%s> computational sequence successfully wrote to %s ..."%(rootName,destination))

def metadata_to_dict(metadata_):
	if (type(metadata_) is dict): 
		return metadata_
	else:
		metadata={}
		for key in metadata_.keys(): 
			try:
				metadata[key]=json.loads(metadata_[key][0])
			except:
				try:
					metadata[key]=str(metadata_[key][0])
				except:
					log.error("Metadata %s is in wrong format. Exiting ...!"%key)
		return metadata


