# Cybersecurity LM 

An inferrer for semi-supervised anomaly detection on multivariate time-series data. 

## Long Short-Term Memory Autoencoder
The idea behind this algorithm is to learn to reconstruct 
a dataset of normal instances. Then the reconstruction error distribution is 
used to define anomaly scores for newly seen data. Anomalous data are expected 
to have a worse reconstruction error and thus, a higher anomaly score. Based on *[Malhotra et al.](https://arxiv.org/abs/1607.00148)*

## Training data 
https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Contact
- mina.marmpena@itml.gr
- spoliti@itml.gr