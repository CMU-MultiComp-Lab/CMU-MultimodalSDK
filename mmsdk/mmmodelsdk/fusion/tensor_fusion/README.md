# Tensor Fusion Networks

This is the code for [Tensor Fusion Network for Multimodal Sentiment Analysis](http://aclweb.org/anthology/D17-1115) published in EMNLP 2017 and presented in multimodal session. 

## FAQ: 

#### Why are results  for TFN  different from TFN paper to say MFN paper?

TFN paper uses 10-fold cross-validation while papers after TFN use folds established by Poria et.a. 2017 (ACL). 

#### Is the fusion algorithm different than bilinear compact pooling?

Yes, bilinear compact pooling is a special case of tensor fusion with only trimodal component. To keep the eigen vectors of each modality (unimodal), and also model bimodal interactions, tensor fusion uses a different tensor operation.

#### Why does TFN work well in practice?

Any machine learning problem has an inductive bias behind it. For example, why do we use CNNs for vision? TFN is great for getting interaction between multimodal dimensions. These interactions are usually sparse and TFN allows everything to efficiently interact with everything else. This way, all information is used, and interactions are more efficiently formed. Surprisingly, due to better modeling of interactions, TFN in practice overfits less than early fusion.
