#scenario1.py
#performs imputations and unifies to make sure they remain functional after changes.

import mmsdk
from mmsdk import mmdatasdk
import numpy
import sys

#uncomment all the ==> lines together

#A simple averaging technique. More advanced methods can be built based on intervals.
def myavg(intervals,features):
        return numpy.average(features,axis=0)

def impute_and_unify_diagnostics():

	cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')

	#removing one video to check the unify()
	some_video=list(cmumosi_highlevel["glove_vectors"].data.keys())[0]
	cmumosi_highlevel["glove_vectors"]._remove_id(some_video)
	cmumosi_highlevel.unify()


	#Aligning to the words to get word-level alignments
	cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])

	#removing one segment to check the unify()
	some_word=list(cmumosi_highlevel["glove_vectors"].data.keys())[0]
	some_word_video=some_word.split("[")[0]
	cmumosi_highlevel["glove_vectors"]._remove_id(some_word)
	cmumosi_highlevel.impute('glove_vectors')
	#removing an entire video using purge to check unify
	cmumosi_highlevel["glove_vectors"]._remove_id(some_word_video,purge=True)
	cmumosi_highlevel.revert()
	return True

if __name__=='__main__':
	impute_and_unify_diagnostics()
	print ("Test finished with python version %s"%str(sys.version_info))

