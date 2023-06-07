from mmsdk.mmdatasdk import log, computational_sequence
import sys
import numpy
import time
from tqdm import tqdm
import os

#specified for numerical inconsistencies within floating points - if abs l1 distance two numbers is less than this, then they are the same. 
#only use for interval comparison.
epsilon=10e-6

class mmdataset:

	def __init__(self,recipe,destination=None):
		self.computational_sequences={}

		if type(recipe) is str:
			if os.path.isdir(recipe) is False:
				log.error("Dataset folder does not exist ...",error=True)

			from os import listdir
			from os.path import isfile, join
			computational_sequence_list = [f for f in listdir(recipe) if isfile(join(recipe, f)) and f[-4:]=='.csd']
			for computational_sequence_fname in computational_sequence_list:
				this_sequence=computational_sequence(join(recipe,computational_sequence_fname))
				self.computational_sequences[this_sequence.metadata["root name"]]=this_sequence

		elif type(recipe) is dict:
			for entry, address in recipe.items():
				self.computational_sequences[entry]=computational_sequence(address,destination)

		elif type(recipe) is list:
			#TODO: computational sequence with uuid
			pass

		else:
			log.error("Cannot create a dataset with the given recipe. Exiting ...!",error=True)

		if len(self.computational_sequences.keys())==0:
			log.error("Dataset failed to initialize ...", error=True)

		log.success("Dataset initialized successfully ... ")

	def __getitem__(self,key):
		if key not in list(self.computational_sequences.keys()):
			log.error("Computational sequence does not exist ...",error=True)
		return self.computational_sequences[key]

	def keys(self):
		return self.computational_sequences.keys()

	def add_computational_sequences(self,recipe,destination):
		for entry, address in recipe.items():
			if entry in self.computational_sequences:
				log.error("Dataset already contains <%s> computational sequence ..."%entry)
			self.computational_sequences[entry]=computational_sequence(address,destination)

	def bib_citations(self,outfile=None):
		outfile=sys.stdout if outfile is None else outfile
		sdkbib='@article{zadeh2018multi, title={Multi-attention recurrent network for human communication comprehension}, author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe}, journal={arXiv preprint arXiv:1802.00923}, year={2018}}'
		outfile.write('mmsdk bib: '+sdkbib+'\n\n')
		for entry,compseq in self.computational_sequences.items():
			compseq.bib_citations(outfile)

	def unify(self,active=True):
		log.status("Unify was called ...")

		all_vidids={}
		violators=[]
		
		all_keys={}
		for seq_key in list(self.computational_sequences.keys()):
			all_keys[seq_key]=[vidid.split("[")[0] for vidid in self.computational_sequences[seq_key].data.keys()]
		
		valids=set.intersection(*[set(all_keys[x]) for x in all_keys])
		violators=set()
		for seq_key in list(self.computational_sequences.keys()):
			violators=violators.union(set([vidid.split("[")[0] for vidid in self.computational_sequences[seq_key].data.keys()])-valids)
		
		if len(violators) >0 :
			for violator in violators:
				log.error("%s entry is not shared among all sequences, removing it ..."%violator,error=False)
				if active==True:
					self.remove_id(violator,purge=True)
		if active==False and len(violators)>0:
			log.error("%d violators remain, alignment will fail if called ..."%len(violators),error=True)

		log.success("Unify completed ...")

	def hard_unify(self,active=True):
		log.status("Hard unify was called ...")

		all_vidids={}
		violators=[]
		
		all_keys={}
		for seq_key in list(self.computational_sequences.keys()):
			all_keys[seq_key]=[vidid for vidid in self.computational_sequences[seq_key].data.keys()]
		
		valids=set.intersection(*[set(all_keys[x]) for x in all_keys])
		for seq_key in list(self.computational_sequences.keys()):
			hard_unify_compatible=all(["[" in vidid for vidid in self.computational_sequences[seq_key].data.keys()])
			if hard_unify_compatible is False:
				log.error("Hard unify can only be done on aligned computational sequences, %s violated this ... Exiting ..."%seq_key)
			violators=set([vidid for vidid in self.computational_sequences[seq_key].data.keys()])-valids
			for violator in violators:
				if active==True:
					log.error("%s entry is not shared among all sequences, removing it ..."%violator,error=False)
					self[seq_key]._remove_id(violator,purge=False)

			if active==False and len(violators)>0:
				log.error("%d violators remain, alignment will fail if called ..."%len(violators),error=True)
		
		log.success("Hard unify completed ...")


	
	def remove_id(self,entry_id,purge=False):
		for key,compseq in self.computational_sequences.items():
			compseq._remove_id(entry_id,purge=purge)

	def align(self,reference,collapse_functions=None,replace=True):
		aligned_output={}

		for sequence_name in self.computational_sequences.keys():
			aligned_output[sequence_name]={}
		if reference not in self.computational_sequences.keys():
			log.error("Computational sequence <%s> does not exist in dataset"%reference,error=True)
		refseq=self.computational_sequences[reference].data
		#unifying the dataset, removing any entries that are not in the reference computational sequence
		self.unify()

		#building the relevant entries to the reference - what we do in this section is simply removing all the [] from the entry ids and populating them into a new dictionary
		log.status("Pre-alignment based on <%s> computational sequence started ..."%reference)
		relevant_entries=self.__get_relevant_entries(reference)
		log.status("Alignment starting ...")

		pbar = log.progress_bar(total=len(refseq.keys()),unit=" Computational Sequence Entries",leave=False)
		pbar.set_description("Overall Progress")
		for entry_key in list(refseq.keys()):
			pbar_small=log.progress_bar(total=refseq[entry_key]['intervals'].shape[0],unit=" Segments",leave=False)
			pbar_small.set_description("Aligning %s"%entry_key)
			for i in range(refseq[entry_key]['intervals'].shape[0]):
				#interval for the reference sequence
				ref_time=refseq[entry_key]['intervals'][i,:]
				#we drop zero or very small sequence lengths - no align for those
				if (abs(ref_time[0]-ref_time[1])<epsilon):
					pbar_small.update(1)
					continue

				#aligning all sequences (including ref sequence) to ref sequence
				for otherseq_key in list(self.computational_sequences.keys()):
					if otherseq_key != reference:
						intersects,intersects_features=self.__intersect_and_copy(ref_time,relevant_entries[otherseq_key][entry_key],epsilon)
					else:
						intersects,intersects_features=refseq[entry_key]['intervals'][i,:][None,:],refseq[entry_key]['features'][i,:][None,:]
					#there were no intersections between reference and subject computational sequences for the entry
					if intersects.shape[0] == 0:
						continue
					#collapsing according to the provided functions
					if type(collapse_functions) is list:
						intersects,intersects_features=self.__collapse(intersects,intersects_features,collapse_functions)
					if(intersects.shape[0]!=intersects_features.shape[0]):
						log.error("Dimension mismatch between intervals and features when aligning <%s> computational sequences to <%s> computational sequence"%(otherseq_key,reference),error=True)
					aligned_output[otherseq_key][entry_key+"[%d]"%i]={}
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"]=intersects
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["features"]=intersects_features
				pbar_small.update(1)
			pbar_small.close()
			pbar.update(1)
		pbar.close()
		log.success("Alignment to <%s> complete."%reference)
		if replace is True:
			log.status("Replacing dataset content with aligned computational sequences")
			self.__set_computational_sequences(aligned_output)
			return None
		else:
			log.status("Creating new dataset with aligned computational sequences")
			newdataset=mmdataset({})
			newdataset.__set_computational_sequences(aligned_output,metadata_copy=False)
			return newdataset

	def __collapse(self,intervals,features,functions):
		#we simply collapse the intervals to (1,2) matrix
		new_interval=numpy.array([[intervals.min(),intervals.max()]])
		try:
			new_features=numpy.concatenate([function(intervals,features) for function in functions],axis=0)
			if len(new_features.shape)==1:
				new_features=new_features[None,:]
		except:
			log.error("Cannot collapse given the set of functions. Please check for exceptions in your functions ...", error=True)
		return new_interval,new_features

	#TODO: This is not passive-align safe
	def revert(self,replace=True):
		reverted_dataset={x:{} for x in self.keys()}
		log.status("Revert was called ...")
		if len(self.keys())==0:
			log.error("The dataset contains no computational sequences ... Exiting!",error=True)

		self.hard_unify()
		all_keys=list(self[list(self.keys())[0]].keys())

		if len(all_keys)==0:
			log.error("No entries in computational sequences or removed during unify ... Exiting!")

		unique_unnumbered_entries={}

		for key in all_keys:
			if key.split('[')[0] not in unique_unnumbered_entries:
				unique_unnumbered_entries[key.split('[')[0]]=[]
			unique_unnumbered_entries[key.split('[')[0]].append(int(key.split('[')[1][:-1]))

		pbar = tqdm(total=len(unique_unnumbered_entries.keys()),unit=" Unique Sequence Entries",leave=False)
		pbar.set_description("Reversion Progress")
		for key in unique_unnumbered_entries.keys():
			unique_unnumbered_entries[key].sort()
			for cs_key in reverted_dataset.keys():
				intervals=numpy.concatenate([self[cs_key][str('%s[%d]'%(key,i))]["intervals"] for i in unique_unnumbered_entries[key]],axis=0)
				features=numpy.concatenate([self[cs_key][str('%s[%d]'%(key,i))]["features"] for i in unique_unnumbered_entries[key]],axis=0)
				reverted_dataset[cs_key][key]={"intervals":intervals,"features":features}
			pbar.update(1)
		pbar.close()
		log.success("Reversion completed ...")
		if replace is True:
			log.status("Replacing dataset content with reverted computational sequences")
			self.__set_computational_sequences(reverted_dataset)
			return None
		else:
			log.status("Creating new dataset with reverted computational sequences")
			newdataset=mmdataset({})
			newdataset.__set_computational_sequences(reverted_dataset,metadata_copy=False)
			return newdataset	

	def impute(self,ref_key,imputation_fn=numpy.zeros):
		log.status("Imputation called with function: %s ..."%str(imputation_fn.__name__))
		other_keys=list(self.keys())
		other_keys.remove(ref_key)
		other_keys_dims={x:list(self[x][list(self[x].keys())[0]]["features"].shape[1:]) for x in other_keys}
		pbar = tqdm(total=len(self[ref_key].keys()),unit=" Reference Computational Sequence Entries",leave=False)
		pbar.set_description("Imputation Progress")
		for seg_key in self[ref_key].keys():
			for other_key in other_keys:
				try:
					self[other_key][seg_key]
				except:
					num_entries=self[ref_key][seg_key]["intervals"].shape[0]
					self[other_key][seg_key]={"intervals":self[ref_key][seg_key]["intervals"],"features":imputation_fn([num_entries]+other_keys_dims[other_key])}
					#log.status("Imputing %s,%s with function %s"%(other_key,seg_key,str(imputation_fn.__name__)))
			pbar.update(1)
		pbar.close()
		log.success("Imputation completed ...")


	#setting the computational sequences in the dataset based on a given new_computational_sequence_data - may copy the metadata if there is already one
	def __set_computational_sequences(self,new_computational_sequences_data,metadata_copy=True):

		#getting the old metadata from the sequence before replacing it. Even if this is a new computational sequence this will not cause an issue since old_metadat will just be empty
		old_metadata={m:self.computational_sequences[m].metadata for m in list(self.computational_sequences.keys())}
		self.computational_sequences={}
		for sequence_name in list(new_computational_sequences_data.keys()):
			self.computational_sequences[sequence_name]=computational_sequence(sequence_name)
			self.computational_sequences[sequence_name].setData(new_computational_sequences_data[sequence_name],sequence_name)
			if metadata_copy:
				#if there is no metadata for this computational sequences from the previous one or no previous computational sequenece
				if sequence_name not in list(old_metadata.keys()):
					log.error ("Metadata not available to copy ..., please provide metadata before writing to disk later", error =False)
				self.computational_sequences[sequence_name].setMetadata(old_metadata[sequence_name],sequence_name)
			self.computational_sequences[sequence_name].rootName=sequence_name

	def deploy(self,destination,filenames):
		if os.path.isdir(destination) is False:
			os.mkdir(destination)
		for seq_key in list(self.computational_sequences.keys()):
			if seq_key not in list(filenames.keys()):
				log.error("Filename for %s computational sequences not specified"%seq_key)
			filename=filenames[seq_key]
			if filename [:-4] != '.csd':
				filename+='.csd'
			self.computational_sequences[seq_key].deploy(os.path.join(destination,filename))

	def sort(self,sort_function=sorted):
		""" Sorts the computational sequence data based on the start time in intervals 

		#Arguments
			self: The dataset object.
			sort_function: The function passed for sorting. This function will receive the intervals one by one for each key 

		#Returns
			Does not return any values - replaces in place

		"""


		for key in list(self.keys()):
			for csd in list(self[key].keys()):
				this_intervals_np=numpy.array(self[key][csd]["intervals"])
				this_features_np=numpy.array(self[key][csd]["features"])
				sorted_indices=sort_function(range(this_intervals_np.shape[0]),key=lambda x: this_intervals_np[x,0])
				sorted_this_intervals_np=this_intervals_np[sorted_indices,:]
				sorted_this_features_np=this_features_np[sorted_indices,:]
				self[key][csd]["intervals"]=sorted_this_intervals_np
				self[key][csd]["features"]=sorted_this_features_np

	#TODO: Add folds to this function
	def get_tensors(self,seq_len,non_sequences=[],direction=False,folds=None):
		""" Returns trainable tensor from computational sequence data

		#Arguments
			self: The dataset object.
			seq_len: The maximum sequence length for the computational sequence entries, e.g. sentence length in words.
			direction: True for right padding and False for left padding.
			folds: The folds in which the dataset should be split.
		#Returns
			Dictionary of numpy arrays with the same data type as computational sequences. Dictionaries include the same keys as the dataset

		"""

		csds=list(self.keys())

		def handle_none_folds():
			return_folds={}
			for key in list(self[csds[0]].keys()):
				return_folds[key.split("[")[0]]=None
			return [list(return_folds.keys())]
	
		if folds==None:
			log.error("No fold specified for get_tensors, defaulting to the first computational sequence ids",error=False)
			folds=handle_none_folds()

		self.hard_unify()

		data=[]
		output=[]
	
		for i in range (len(folds)):
			data.append({})
			output.append({})
		
		def lpad(this_array,direction):
			if direction==False:
				temp_array=numpy.concatenate([numpy.zeros([seq_len]+list(this_array.shape[1:])),this_array],axis=0)[-seq_len:,...]
			else:
				temp_array=numpy.concatenate([this_array,numpy.zeros([seq_len]+list(this_array.shape[1:]))],axis=0)[:seq_len,...]
			return temp_array

		def detect_entry_fold(entry,folds):
			entry_id=entry.split("[")[0]
			for i in range(len(folds)):
				if entry_id in folds[i]: return i
			return None
			

		if len(csds)==0:
			log.error("Dataset is empty, cannot get tensors. Exiting ...!",error=True)

		for i in range (len(folds)):
			for csd in csds:
				data[i][csd]=[]

		for key in list(self[csds[0]].keys()):
			which_fold=detect_entry_fold(key,folds)
			if which_fold==None:
				log.error("Key %s doesn't belong to any fold ... "%str(key),error=False)
				continue
			for csd in list(self.keys()):
				this_array=self[csd][key]["features"]
				if csd in non_sequences:
					data[which_fold][csd].append(this_array)
				else:
					data[which_fold][csd].append(lpad(this_array,direction))

		for i in range(len(folds)):
			for csd in csds:
				output[i][csd]=numpy.array(data[i][csd])

		return output

	def __intersect_and_copy(self,ref,relevant_entry,epsilon):

		sub=relevant_entry["intervals"]
		features=relevant_entry["features"]

		#copying and inverting the ref
		ref_copy=ref.copy()
		ref_copy[1]=-ref_copy[1]
		ref_copy=ref_copy[::-1]
		sub_copy=sub.copy()
		sub_copy[:,0]=-sub_copy[:,0]
		#finding where intersect happens
		where_intersect=(numpy.all((sub_copy-ref_copy)>(-epsilon),axis=1)==True)
		intersectors=sub[where_intersect,:]
		intersectors=numpy.concatenate([numpy.maximum(intersectors[:,0],ref[0])[:,None],numpy.minimum(intersectors[:,1],ref[1])[:,None]],axis=1)
		intersectors_features=features[where_intersect,:]
		#checking for boundary cases and also zero length
		where_nonzero_len=numpy.where(abs(intersectors[:,0]-intersectors[:,1])>epsilon)
		intersectors_final=intersectors[where_nonzero_len]
		intersectors_features_final=intersectors_features[where_nonzero_len]
		return intersectors_final,intersectors_features_final

	#TODO: Need tqdm bar for this as well
	def __get_relevant_entries(self,reference):
		relevant_entries={}
		relevant_entries_np={}

		#pbar = tqdm(total=count,unit=" Computational Sequence Entries",leave=False)


		for otherseq_key in set(list(self.computational_sequences.keys()))-set([reference]):
			relevant_entries[otherseq_key]={}
			relevant_entries_np[otherseq_key]={}
			sub_compseq=self.computational_sequences[otherseq_key] 
			for key in list(sub_compseq.data.keys()):              
				keystripped=key.split('[')[0]                  
				if keystripped not in relevant_entries[otherseq_key]:                           
					relevant_entries[otherseq_key][keystripped]={}
					relevant_entries[otherseq_key][keystripped]["intervals"]=[]                     
					relevant_entries[otherseq_key][keystripped]["features"]=[]                                                            
		        
				relev_intervals=self.computational_sequences[otherseq_key].data[key]["intervals"]                                             
				relev_features=self.computational_sequences[otherseq_key].data[key]["features"]         
				if len(relev_intervals.shape)<2:
					relev_intervals=relev_intervals[None,:]
					relev_features=relev_features[None,:]

				relevant_entries[otherseq_key][keystripped]["intervals"].append(relev_intervals)
				relevant_entries[otherseq_key][keystripped]["features"].append(relev_features)
		                        
			for key in list(relevant_entries[otherseq_key].keys()):
				relev_intervals_np=numpy.concatenate(relevant_entries[otherseq_key][key]["intervals"],axis=0)                                 
				relev_features_np=numpy.concatenate(relevant_entries[otherseq_key][key]["features"],axis=0)
				sorted_indices=sorted(range(relev_intervals_np.shape[0]),key=lambda x: relev_intervals_np[x,0])                               
				relev_intervals_np=relev_intervals_np[sorted_indices,:]                         
				relev_features_np=relev_features_np[sorted_indices,:]

				relevant_entries_np[otherseq_key][key]={}
				relevant_entries_np[otherseq_key][key]["intervals"]=relev_intervals_np
				relevant_entries_np[otherseq_key][key]["features"]=relev_features_np
			log.status("Pre-alignment done for <%s> ..."%otherseq_key)
		return relevant_entries_np
