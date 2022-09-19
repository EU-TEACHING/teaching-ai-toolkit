# Anomaly Detection LM for Cybersecurity

A trainer and an inferrer for semi-supervised anomaly detection on multivariate time-series data. 

## Long Short-Term Memory Autoencoder
The idea behind this algorithm is to learn to reconstruct 
a dataset of normal instances. Then the reconstruction error distribution is 
used to define anomaly scores for newly seen data. Anomalous data are expected 
to have a worse reconstruction error and thus, a higher anomaly score. Based on *[Malhotra et al.](https://arxiv.org/abs/1607.00148)*

The network works in two execution modes: training and inference.

### Training mode
###### Input
- The paths of the dataframes (normal and anomalous), as well as other 
  training parameters, are declared in the configuration file:

    `src/configs/config.py` 

- The mode is declared as `--exec 'train` or `-e 'train'`. 


### Inference mode
###### Input
- The mode is declared as `--exec 'train` or `-e 'infer'`. 
- - The paths of the dataframes (normal and anomalous), as well as other 
  training parameters, are declared in the configuration file:
`src/configs/config.py` 

###### Output
Stored in: `teaching/learning_modules/cybersecurity/local_results_storage`

A dataframe with the initial features expanded by a column 'pred_label' 
indicating normal (0) or anomalous (1) samples according to the 
anomaly detection algorithm.

Run from the root directory with:


    python main.py -e 'infer'