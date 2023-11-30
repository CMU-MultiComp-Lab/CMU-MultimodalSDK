# CMU-Multimodal SDK Version 1.2.0 (mmsdk)

CMU-Multimodal SDK provides tools to easily load well-known multimodal datasets and rapidly build neural multimodal deep models. Hence the SDK comprises of two modules: 1) mmdatasdk: module for downloading and procesing multimodal datasets using computational sequences. 2) mmmodelsdk: tools to utilize complex neural models as well as layers for building new models. The fusion models in prior papers will be released here. 

All the datasets here are processed using the SDK (even the old_processed_data folder which uses SDK V0). You can acquire the citations for the computational sequences used in your project by calling the below functions on your dataset:	

```python
>>> mydataset.bib_citations(open('mydataset.bib','w'))
>>> mycompseq.bib_citations(open('mycompseq.bib','w'))
```

If you use the data and models created under the CMU Multimodal SDK, please consider citing the research paper that sparked the creation of the SDK and unified multiple multimodal datasets:

``` 
@inproceedings{zadeh2018multi,
  title={Multi-attention recurrent network for human communication comprehension},
  author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

If you are only interested in downloading the word-aligned data and models to run experiments on (and not interested in exploring new features or alignment techniques), the ready-to-go aligned files can be downloaded from: 

Data for MOSI and MOSEI: https://github.com/pliang279/MultiBench

RL, MFN, MARN, Graph-MFN Model Theano Codes (theano is now deprecated, please refer to related repos for pytorch implementation for some of these models): https://drive.google.com/drive/u/2/folders/1NxyFuogyzNFoCH0Zi5aIXUGYKGuSY9TY

# News

--> New features created for POM by Ebenge Usip: https://github.com/eusip/POM/

--> We will be releasing the **MOSEAS-WE1** (Multimodal Opinion Sentiment, Emotions and Attributes - West Europe V1) Dataset on November 15th 2020. The dataset is similar to MOSEI, except for languages of Spanish, French, German and Portuguese. We also include attributes similar to POM dataset on top of emotions and sentiment. 

--> Version 1.2.0 released. Allowing for compression, and improving the computational sequence modules. 

--> **Code for Factorized Multimodal Transformer** is released here (current SOTA on MOSI, IEMOCAP minus happy, POM): https://github.com/A2Zadeh/Factorized-Multimodal-Transformer

--> **Version 1.1.1 announced**. Social-IQ public test set has been added. Some new functionalities are also added including hard_unify. Some previous functionalities including unify and impute are improved. 

--> **We released all the previous code which was in theano - check FAQ for download**. RL for Sentiment, MFN, MARN, and Graph-MFN. Unfortunately theano got deprecated before we could clean up these codes for release, and we moved on to pytorch. The codes are exactly identical to what we submitted the papers with, and they are running with the latest version available through pip. Newest pytorch implementations of our fusion methods are available in mmmodelsdk in mmsdk. 

--> **Version 1.1 announced**. Social-IQ 1.0 has been released. 

--> **Version 1.0.5 announced**. Two new functions got added: revert and imputate. Revert takes an aligned computational sequence with segment ids (i.e. y[z]) and builds a computational sequence without segment ids (i.e. y) by stacking intervals and features. Imputate fills the gaps in alignment, say computational sequence x does not have segment y[z] then that segment can be filled with say zeros with the same shape as required by x. A reference computational sequence is needed for this purpose to provide intervals. The imputation feature shape will be (1 by shape of x featureset). These two functions take a lot of guess work out of the equation when developing models. Tutorials coming soon :)

--> A nice tutorial on how to use the SDK is prepared by Zhun Liu [here](https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials). SDK should not be considered trivial to use. The machine learning scientists are still involved in a lot of decisions e.g. how to handle cases where some csd has time gaps (like detector fails or nan values). 

--> **Version 1.0.4 announced**. Special thanks to Zhun Liu and Michael Grundie. New features added to CMU-MOSI, CMU-MOSEI and POM datasets. TLDR: Word alignment on CMU-MOSI is substantially improved, therefore the language modality is now working much better - previous alignment for MOSI is now deprecated. This new alignment is not applied to GloVe and BERT, only to the raw words - **hence I removed the glove and BERT for now - if you decide to build your own BERT you should consider running BERT on each segment not on the entire video since it is a contextual word embedding.** New features including OpenFace2, OpenSmile-IS09 are also added. Computational sequences for IEMOCAP are also released (only available by proof of email from USC, the zip file is password protected). Please check the next_steps.md to see where we are going next.

--> Alignment function on large datasets improved ~40x in speed. CMU-MOSEI now aligns in less than 4 hours. Previously the full dataset took around 2-3 days to fully align, majority of which was spent on alignment function.

--> Have a look at the newly released RAVEN model: https://github.com/victorywys/RAVEN - https://arxiv.org/pdf/1811.09362.pdf - tldr: while previously we averaged nonverbal information for each word under assumption that subword nonverbal behaviors are probably mostly constant, we recently discovered that better modeling of subword nonverbal behaviors actually helps a lot! More than we originally anticipated, we are able to achieve competative results with SOTA just using a LSTM on nonverbal shifts in word vectors.

## CMU Multimodal Data SDK (mmdatasdk)

CMU-Multimodal Data SDK simplifies downloading and loading multimodal datasets. The module mmdatasdk treats each multimodal dataset as a combination of **computational sequences**. Each computational sequence contains information from one modality in a heirarchical format, defined in the continuation of this section. Computational sequences are self-contained and independent; they can be used to train models in isolation. They can be downloaded, shared and registered with our trust servers. This allows the community to share data and recreate results in a more elegant way using computational sequence intrgrity checks. Furthermore, this integrity check allows users to download the correct computational sequences. 

Each computational sequence is a heirarchical data strcuture which contains two key elements 1) "data" is a heirarchy of features in the computational sequence categorized based on unique multimodal source identifier (for example video id). Each multimodal source has two matrices associated with it: features and intervals. Features denote the computational descriptors and intervals denote their associated timestamp. Both features and intervals are numpy 2d arrays. 2) "metadata": contains information about the computational sequence including integrity and version information. The computational sequences are stored as hdf5 objects on hard disk with ".csd" extension (computational sequential data). Both the data and metadata are stored under "root name" (root of the heirarchy)

A dataset is defined as a dictionary of multiple computational sequences. Entire datasets can be shared using recipes as opposed to old-fashioned dropbox links or ftp servers. Computational sequences are downloaded one by one and their individual integrity is checked to make sure they are the ones users wanted to share. Users can register their extracted features with our trust server to use this feature. They can also request storage of their features on our servers 


## 🚀 Installation

The first step is to download the SDK:

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalSDK.git
```

Next, you need to install the SDK on your python enviroment.

```bash
cd CMU-MultimodalSDK
pip install .
```
This will install the mmsdk module and all the required dependencies on your python enviroment and you can start using it.

You can also install the SDK in development mode:

```bash
cd CMU-MultimodalSDK
pip install -e .
```


## Usage

The first step in most machine learning tasks is to acquire the data. We will work with CMU-MOSI for this readme. 

```python
>>> from mmsdk import mmdatasdk
```

Now that mmdatasdk is loaded you can proceed to fetch a dataset. The datasets are a set of computational sequences, where each computational sequence hosts the information from a modality or a view of a modality. For example a computational sequence could be the word vectors and another computational sequence could be phoneme 1-hot vectors. 

If you are using a standard dataset, you can find the list of them in the mmdatasdk/dataset/standard_datasets. We use CMU-MOSI for now. We will work with highlevel features (glove embeddings, facet facial expressions, covarep acoustic features, etc)

```python
>>> from mmsdk import mmdatasdk
>>> cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
```

This will download the data using the links provided in *mmdatasdk.cmu_mosi.highlevel* dictionary (mappings between computational sequence keys and their respective download link) and put them in the *cmumosi/* folder. 

The data that gets downloaded comes in different frequencies, however, they computational sequence keys will always be the same. For example if video v0 exists in glove embeddings, then v0 should exist in other computational sequences as well. The data with different frequency is applicable for machine learning tasks, however, sometimes the data needs to be aligned. The next stage is to align the data according to a modality. For example we would like to align all computational sequences according to the labels of a dataset. First, we fetch the opinion segment labels computational sequence for CMU-MOSI. 

```python
>>> cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
```

Next we align everything to the opinion segment labels. 

```python
>>> cmumosi_highlevel.align('Opinion Segment Labels')
```

*Opinion Segment Labels* is the key for the labels we just fetched. Since every video has multiple segments according to annotations and timing in opinion segment labels, each video will also be accompanied by a [x] where x denotes which opinion segment the computational sequence information belongs to; for example v0[2] denotes third segment of v0 (starting from [0]). 


**Word Level Alignement:**

In recent papers, it has been a common practice to perform word-level alignment. To do this with the mmdatasdk, we can do the following:

```python
>>> from mmsdk import mmdatasdk
>>> cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
>>> cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])
>>> cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
>>> cmumosi_highlevel.align('Opinion Segment Labels')
```

we first aligned everything to the *glove_vectors* modality and then we align to the *Opinion Segment Labels*. Please note that with the alignment to the *glove_vectors*, we ask the align function to also collapse the other modalities. This basically means summarize the other modalities based on a set of functions. The functions all receive two argument *intervals* and *features*. Intervals is a *m times 2* and features is a *m times n* matrix. The output of the functions should be *1 times n*. For example the following function ignores intervals and just takes the average of the input features:

```python
import numpy
def myavg(intervals,features):
        return numpy.average(features,axis=0)
```

Multiple functions can be passed to *collapse_functions*, each of them will be applied one by one and will be concatenated as the final output. 


# Supported Datasets

**Social-IQ 1.0**: Social-IQ 1.0 is a benchmark dataset for artificial social intelligence. It involves questions and answers for online social media videos where natuarlistic interactions happen between involved individuals in the video. 

**CMU-MOSEI**: CMU-MOSEI contains a large number of in-the-wild videos annotated for sentiment and emotions. The annotations follow consensus-based online perceptions. Alongside the value of this dataset for sentiment and emotion recognition, it is also suitable for representation learning due to large size of the dataset. 

**CMU-MOSI**: CMU-MOSI is a standard benchmark for multimodal sentiment analysis. It is specially suited to train and test multimodal models, since most of the newest works in multimodal temporal data use this dataset in their papers. 

**POM**: POM is a dataset for analysis of persuasion on online social media. It has annotations for personality and sentiment as well, which makes it very compelling for large number of tasks. 

# Frequently Asked Questions

#### 1. What does the SDK do for me? 

CMU Multimodal SDK helps standardize data sharing and multimodal modeling for the field of multimodal machine learning. It provides functionalities not available in other resources. These functionalities include alignment and neural models that are not available in other libraries. 

#### 2. SDK keeps developing, how do I keep track?

Check for the version of the SDK and make sure you are up to date. Furthermore, try to pull the code every now and then. 

#### 3. Computational sequences keep getting more and more advanced, how do I keep up?

The goal of SDK is to make sure users always have access to the state of the art. Therefore, results may improve naturally over time, just because you have access to better tools. Check the version of the computational sequences to make sure you are using the most recent ones. If you pull the SDK and download the computational sequences from standard datasets, you will always download the most recent. 

#### 4. Do you share raw data?

No - we are not allowed to share the raw videos due to privacy of Youtube content creators. We only share anonymized processed features.

#### 5. How do I check my results with previous work, if computational sequences keep on improving?

It is a reasonable expectation that a researcher should dedicate some time to recreating previous works. However, with the new computational sequences, some models may improve naturally. You should run the previous works on the newest data always, if your approach uses the newest data. The implementations for most of the previous works are either available through auhtors works (ours is available within mmmodelsdk). 

#### 6. Can I add computational sequences or add a new dataset to the SDK?

Of course, mmdatasdk is built for that purpose. We will release tutorials on how to release your own dataset. If you are familiar with the format of computational sequences, that should be relatively easy to do. Furthermore, we are also implementing hasing checks to allow researchers to register their data, and simply share the hash to allow others to recreate their experiments reliably. 

#### 7. What about the old implementation of the SDK? 

Through the very first implementation of the SDK, we learned a lot about necessary optimizations and how to manage the datasets. At this point in time the old version of SDK is completely deprecated. Please use the new SDK, called CMU Multimodal SDK (you are in the correct gituhb). 

#### 8. I found an issue in the code or in a computational sequence, what do I do?

Please report it ASAP on the issues tab in github or simply contact us through email. We normally respond within 12 hours to the most urgent issues. 

#### 9. Do you share the code and data for publications prior to mid 2018?

Yes, we do, here is the code (theano): https://drive.google.com/drive/u/2/folders/1NxyFuogyzNFoCH0Zi5aIXUGYKGuSY9TY. These are exact data used for our experiments, already aligned at word level. You can certainly use this data, but I do advocate exploring the datasets using the SDK. For example try different alignments, or strategies. (Please note that CMU-MOSEI had some issues for some videos over their acoustic modality. They are now solved and CMU-MOSEI downloaded from SDK gets better performance than the one we ran experiments on for original paper)


