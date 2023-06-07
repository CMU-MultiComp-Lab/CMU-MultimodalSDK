**Our next steps for the Multimodal SDK**

**Computational Sequence**

+ Refactoring some of the code
++ Loose function cleanups
+ Adding compression
++ Adding the chunks on hdd
+ Adding json metadata
+ Allow in-hdd access

**Data Processing**

+ Dimension names for computational sequences - done - doing final checks before commit
+ BERT, and GloVe (updates) for CMU-MOSI, CMU-MOSEI, POM - delayed due to better alignment now available
+ Adding passive alignment for RAVEN style models
+ Adding numpy style key search [x,1,2] => video x segment number 2, fine segment number 3

**Alignment Functionalities**

+ A faster implementation of unify based on muliple set intersection
+ Intermediate goal: accomodate low-ram setups by forcing the h5py to remain on the hdd <= done, needs final testing as it broke the alignment + setitem and remove items

**High priority fixes**
None ATM
