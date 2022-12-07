# -*- coding: utf-8 -*-
""" main.py """

# standard
import argparse
import time
import logging

# external
import mlflow

# internal
from modules.cybersecurity.src.configs.config import CFG
from modules.cybersecurity.src.models.lstm_ae import LSTMAutoencoder
from modules.cybersecurity.src.inference.inferrer import Inferrer
from modules.cybersecurity.src.utils.logging_utils import mlflow_config

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')


def run_training():
    """Builds model, loads data, trains and evaluates"""
    with_mlflow = CFG["mlflow_config"]["enabled"]

    if with_mlflow:
        mlflow_run = mlflow_config(CFG["mlflow_config"])

    model = LSTMAutoencoder(CFG)

    load_start = time.time()
    model.load_data()
    print(f'Data load runtime: {time.time() - load_start}')

    model.build()

    train_start = time.time()
    model.train()
    model.eval()
    print(f'Train, eval runtime: {time.time() - train_start}')

    if with_mlflow:
        # Saves model and writes logs to mlflow
        log_start = time.time()
        model.save_model()
        print(f'MLflow logging time: {time.time() - log_start}')
    else:
        model.save_model()

    if mlflow.active_run():
        mlflow.end_run()


def run_inference():
    """Loads a trained model and data for prediction
        :return Dataframe with all the features plus a prediction column: 0=normal, 1=anomaly
    """
    infer = Inferrer(CFG)
    infer.load_model()
    infer.load_data()
    infer.predict()
    infer.eval()


if __name__ == '__main__':

    # This is different now, the data are defined in the src/configs/config.py
    # python teaching/learning_modules/anomaly_detection/main.py -e 'train'
    # python teaching/learning_modules/anomaly_detection/main.py -e 'infer' -m "20210708-114002_lstmae" -d "data/unsw-nb15/attack_short.csv"

    parser = argparse.ArgumentParser(description='Define mode of LM execution and parameters')

    parser.add_argument('-e', '--exec', help="mode of execution: 'train' or 'infer'", required=False, default='infer')
    # parser.add_argument('-m', '--model', help="the name of the trained model, if --exec='infer'", required=False,
    #                     default="20210714-123413_lstmae")
    # parser.add_argument('-d', '--datapath', help="the path to the dataframe, if --exec='infer'", required=False,
    #                     default="data/unsw-nb15/attack_short.csv")

    args = parser.parse_args()

    exec_mode = args.exec

    if exec_mode == 'train':
        run_training()
    elif exec_mode == 'infer':
        run_inference()
