# -*- coding: utf-8 -*-
"""Inferrer"""

# standard
import os
import sys
import joblib
import pandas as pd

# internal
from src.dataloader.dataloader import DataLoader
from src.utils.eval_utils import pred_eval, anomaly_scoring, get_eval_metrics
from src.utils.preprocessing_utils import create_sequences, create_dataframe_of_predicted_labels, transform_df
from src.configs.config import CFG
from src.inference.base_inferrer import BaseInferrer

# external
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras


class Inferrer(BaseInferrer):
    def __init__(self, config):
        super().__init__(config)

        self.model_path = self.config.inference.model_path
        self.transformer_path = self.config.inference.transformer_path
        self.ground_truth_cols = self.config.inference.ground_truth_cols
        self.data_types = self.config.data.data_types
        self.n_rows = self.config.data.n_rows
        self.features = self.config.data.features
        self.threshold = None

        self.data = None
        self.X = None
        self.y = None

        self.model = None
        self.transformer = None

        self.data_pred = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            print("The model was not found")
            sys.exit()

        self.model = keras.models.load_model(self.model_path)
        self.transformer = self._load_transformer(self.transformer_path)

    @staticmethod
    def _load_transformer(transformer_path):
        """ Load transformer. """

        if not os.path.exists(transformer_path):
            print("The dataset was not found")
            sys.exit()
        else:
            transformer = joblib.load(transformer_path)

        return transformer

    def load_data(self):
        columns = self.features + self.ground_truth_cols
        dict_data_types = (dict(zip(self.features, self.data_types)) if self.data_types else None)
        self.data = DataLoader().load_data(columns, self.config.inference, dict_data_types, self.n_rows, 'inference')

        self.X = self.data.loc[:, self.features]
        if self.ground_truth_cols:
            self.y = self.data.loc[:, self.ground_truth_cols]

        self._preprocess()

    def _preprocess(self):
        X_transformed = transform_df(self.X, self.transformer)
        self.X_transformed_seq = create_sequences(X_transformed, self.transformer.seq_time_steps)
        self.y_transformed_seq = create_sequences(self.y, self.transformer.seq_time_steps)

    def predict(self):
        """Predicts results for the verification dataset."""

        mean = self.transformer.mahalanobis_params['mean']
        cov = self.transformer.mahalanobis_params['cov']
        scores = "mahalanobis"
        threshold = self.transformer.threshold

        # Test anomaly subset
        a_accuracy, a_precision, a_recall, a_f1 = get_eval_metrics(self.model,
                                                                   self.X_transformed_seq,
                                                                   self.y_transformed_seq,
                                                                   threshold,
                                                                   self.transformer.seq_time_steps,
                                                                   cov, mean)

        # test_reconstruction_error = self.model.predict(self.X_transformed_seq) - self.X_transformed_seq
        #
        # anomalous_data_indices, _ = anomaly_scoring(test_reconstruction_error, self.X_transformed_seq,
        #                                             self.transformer.seq_time_steps, scores, cov,
        #                                             mean, self.threshold)
        #
        # y_pred = create_dataframe_of_predicted_labels(self.X, anomalous_data_indices)
        # self.data_pred = pd.concat([self.data, y_pred], axis=1)
        #
        # # Print evaluation metrics for batch prediction with ground truth column
        # if self.ground_truth_cols:
        #     self._eval()
        #
        # print(self.data_pred.head(10))
        # return self.data_pred

    def predict__(self):
        """Predicts results for the verification dataset."""
        if self.transformer.mahalanobis_params is not None:
            mean = self.transformer.mahalanobis_params['mean']
            cov = self.transformer.mahalanobis_params['cov']
            scores = "mahalanobis"

        else:
            cov = None
            mean = None
            scores = "mae_loss"
            self.threshold = self.transformer.mae_loss_threshold

        test_reconstruction_error = self.model.predict(self.X_transformed_seq) - self.X_transformed_seq

        anomalous_data_indices, _ = anomaly_scoring(test_reconstruction_error, self.X_transformed_seq,
                                                    self.transformer.seq_time_steps, scores, cov,
                                                    mean, self.threshold)

        y_pred = create_dataframe_of_predicted_labels(self.X, anomalous_data_indices)
        self.data_pred = pd.concat([self.data, y_pred], axis=1)

        # Print evaluation metrics for batch prediction with ground truth column
        if self.ground_truth_cols:
            self._eval()

        print(self.data_pred.head(10))
        return self.data_pred

    def _eval(self):
        accuracy, precision, recall, f1 = pred_eval(self.y, self.data_pred.iloc[:, -1])


if __name__ == '__main__':
    infer = Inferrer(CFG)
    infer.load_model()
    infer.load_data()
    res = infer.predict()
