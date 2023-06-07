CMU-Multimodal Opinion Sentiment Intensity (MOSI) dataset. 

1. The dataset contains 93 videos across 89 distinct speakers. 
2. There is only one person speaking within each video. 
3. The topic of all videos is movie reviews. 
4. The dataset is approximately gender balanced.
5. The sentiment score is between -3,3 (extremely neg, extremely pos)
6. The sentiment binarization in all of the previous papers follows negative/non-negative binary classification with criteria of sentiment score <0 being negative and otherwise being non-negative. Newer approaches (after 2019) leave values equal to zero outside (therefore <0 being negative and >0 being strictly positive). There is around 2-3% difference between these two measures, latter being higher in accuracy.
7. For further details of the dataset, please refer to: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7742221
8. For 7-class sentiment, we use the round function to map to discretized range [-3,3]

# Scoreboard
Lower in this list is better. Feel free to push your novel work! B means language model is BERT. 

| Models       | MAE   | Corr  | Acc-2 | Acc-2 | F-Score | F-Score | Acc-7 |
|--------------|-------|-------|-------|-------|---------|---------|-------|
| BC-LSTM      | 1.079 | 0.581 | 73.9  | -/    | 73.9    | -/      | 28.7  |
| MV-LSTM      | 1.019 | 0.601 | 73.9  | -/    | 74      | -/      | 33.2  |
| TFN (EMNLP17)    | 0.97  | 0.633 | 73.9  | -/    | 73.4    | -/      | 32.1  |
| MARN (AAAI18)        | 0.968 | 0.625 | 77.1  | -/    | 77      | -/      | 34.7  |
| MFN (AAAI18)         | 0.965 | 0.632 | 77.4  | -/    | 77.3    | -/      | 34.1  |
| LMF (ACL18)         | 0.912 | 0.668 | 76.4  | -/    | 75.7    | -/      | 32.8  |
| CH-Fusion    | -     | -     | 80    | -/    | -       | -/      | -     |
| MFM (ICLR19)         | 0.951 | 0.662 | 78.1  | -/    | 78.1    | -/      | 36.2  |
| RAVEN (AAAI19)       | 0.915 | 0.691 | 78    | -/    | 76.6    | -/      | 33.2  |
| RMFN (EMNLP18)        | 0.922 | 0.681 | 78.4  | -/    | 78      | -/      | 38.3  |
| MCTN (AAAI19)        | 0.909 | 0.676 | 79.3  | -/    | 79.1    | -/      | 35.6  |
| CIA          | 0.914 | 0.689 | 79.8  | -/    | -/      | 79.5    | 38.9  |
| HFFN (ACL19)        | -     | -     | -/    | 80.2  | -/      | 80.3    | -     |
| HPFN (NeurIPS19)        | 0.945     | 0.672    | 77.5/    | 77.4  | -/      | -    | 36.9     |
| LMFN         | -     | -     | -/    | 80.9  | -/      | 80.9    | -     |
| DFF-ATMF (B) | -     | -     | -/    | 80.9  | -/      | 81.2    | -     |
| ARGF         | -     | -     | -/    | 81.4  | -/      | 81.5    | -     |
| MulT (ACL20)        | 0.871 | 0.698 | -/    | 83    | -/      | 82.8    | 40    |
| TFN (B)  (EMNLP17)    | 0.901 | 0.698 | -/    | 80.8  | -/      | 80.7    | 34.9  |
| LMF (B)  (ACL18)    | 0.917 | 0.695 | -/    | 82.5  | -/      | 82.4    | 33.2  |
| MFM (B)  (ICLR19)    | 0.877 | 0.706 | -/    | 81.7  | -/      | 81.6    | 35.4  |
| ICCN (B)     | 0.86  | 0.71  | -/    | 83    | -/      | 83      | 39    |
| FMT          | 0.837 | 0.744 | 81.5  | 83.5  | 81.4    |         |       |
| MISA (B)     | 0.783 | 0.761 | 81.8  | 83.4  | 81.7    | 83.6    | 42.3  |
| MAG-BERT (ACL20)   |0.712 | 0.796 | 84.2/86.1  | 84.1/86.0  | -   | -    | 42.3  |
| MAG-XLNet (ACL20)   |0.675 | 0.821 | 85.7/87.9  | 85.6/87.9  | -   | -    | 42.3  |
