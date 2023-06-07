CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU-MOSEI) is the largest dataset of sentence level sentiment analysis and emotion recognition in online videos. CMU-MOSEI contains more than 65 hours of annotated video from more than 1000 speakers and 250 topics. The dataset was introduced in the 2018 Association for Computational Linguistics 2018 and used in the co-located First Grand Challenge and Workshop on Human Multimodal Language. 

CMU-MOSEI has sentiment and 6 emotion scores for sentences in YouTube monologue videos. Multiple emotions can be present at a time. Sentiment is in range [-3,3] and emotions in range [0,3]. Labels are in "All Labels" computational sequence, with order as: [sentiment,happy,sad,anger,surprise,disgust,fear]

Binarization: 

For 7-class sentiment, the following binarization is used:

```def cmumosei_round(a):
        if a < -2:
                res = -3
        if -2 <= a and a < -1:
                res = -2
        if -1 <= a and a < 0:
                res = -1
        if 0 <= a and a <= 0:
                res = 0
        if 0 < a and a <= 1:
                res = 1
        if 1 < a and a <= 2:
                res = 2
        if a > 2:
                res = 3
```

For binary sentiment, we use two metrics : 

1) negative/non-negative: simply <0 or >= 0 (used for papers prior to 2019)
2) negative/positive: <0 or >0 (used for papers after 2019).

Both measures are reported in papers after 2019. 

For binary emotions, we have emotion=0 (emotion not present) and emotion >0 (emotion present).
