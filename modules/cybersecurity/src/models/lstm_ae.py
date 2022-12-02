# -*- coding: utf-8 -*-
"""LSTM-AE model"""

# standard
import joblib
import os
import time
from urllib.parse import urlparse

# internal
from modules.cybersecurity.src.models.base_model import BaseModel
from modules.cybersecurity.src.dataloader.dataloader import DataLoader
from modules.cybersecurity.src.utils.preprocessing_utils import create_sequences, transform_df, main_transformer, \
    create_dataframe_of_predicted_labels
from modules.cybersecurity.src.utils.eval_utils import *
from modules.cybersecurity.src.utils.logging_utils import *
from modules.cybersecurity.src.utils.tuning_utils import model_spaces

# external
import pandas as pd
import tensorflow
import mlflow
from mlflow.models.signature import infer_signature
from tensorflow import keras
from keras.callbacks import EarlyStopping
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin, space_eval
from numpy.random import seed

seed(1)
tensorflow.random.set_seed(2)


class LSTMAutoencoder(BaseModel):
    """LSTM-AE model"""

    def __init__(self, config):
        super().__init__(config)

        self.features = self.config.data.features
        self.data_types = self.config.data.data_types
        self.n_rows = self.config.data.n_rows
        self.ground_truth_cols = self.config.data.ground_truth_cols

        self.normal = None
        self.anomaly = None
        self.normal_y = None
        self.anomaly_y = None

        self.transformer = None

        self.tuning = self.config.tune.tuning
        self.max_evals = self.config.tune.max_evals
        self.hyperparams = None
        self.trials = None

        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.val_subsplits = self.config.train.val_subsplits
        self.units = self.config.train.units
        self.dropout_rate = self.config.train.dropout_rate
        self.loss = self.config.train.loss
        self.seq_time_steps = self.config.train.seq_time_steps
        self.model_storage = self.config.model.storage
        self.stopped_iteration = None
        self.training_runtime = None
        self.model_history = None
        self.base_model = None
        self.model = None

        self.train_reconstruction_error = None
        self.test_reconstruction_error = None

        self.scores = self.config.anomaly_scoring.scores
        self.threshold = None
        self.cov = None
        self.mean = None

        self.datestr = time.strftime("%Y%m%d-%H%M%S")
        self.with_mlflow = self.config.mlflow_config.enabled

    def load_data(self):
        """Loads and Preprocess data """
        columns = self.features + self.ground_truth_cols
        dict_data_types = (dict(zip(self.features, self.data_types)) if self.data_types else None)
        self.normal, self.anomaly = DataLoader().load_data(columns, self.config.data, dict_data_types, self.n_rows,
                                                           mode='train')

        # If there are ground truth columns, remove and store them
        if self.ground_truth_cols is not None:
            self.normal_y = self.normal.loc[:, self.ground_truth_cols] # only keep label as ground truth col, skip attack_cat
            self.anomaly_y = self.anomaly.loc[:, self.ground_truth_cols]
            self.normal = self.normal.loc[:, self.features]
            self.anomaly = self.anomaly.loc[:, self.features]

        self._preprocess_data()

    def _preprocess_data(self):
        """ Splits into training and obs and set training parameters"""

        # Fit transformer to the concatenated data
        self.total = pd.concat([self.normal, self.anomaly])
        self.transformer = main_transformer(self.total)

        # Transform the normal and anomaly data
        self.normal = transform_df(self.normal, self.transformer)
        self.anomaly = transform_df(self.anomaly, self.transformer)

        # Create sequences: normal feats transformed
        self.normal_seq = create_sequences(self.normal, self.seq_time_steps)
        print("Normal data input shape (#seqs, seq len, feats): ", self.normal_seq.shape)

        # Create sequences: anomaly feats transformed
        self.anomaly_seq = create_sequences(self.anomaly, self.seq_time_steps)
        print("Anomaly data input shape: ", self.anomaly_seq.shape)

        # Create sequences: normal y
        self.normal_y_seq = create_sequences(self.normal_y, self.seq_time_steps)
        print("Normal target shape (#seqs, seq len", self.normal_y_seq.shape)

        # Create sequences: anomaly y
        self.anomaly_y_seq = create_sequences(self.anomaly_y, self.seq_time_steps)
        print("Anomaly target shape (#seqs, seq len): ", self.anomaly_y_seq.shape)

        self._split_subsets()

    def _split_subsets(self):
        n_seq_num = self.normal_seq.shape[0] // 4
        a_seq_num = self.anomaly_seq.shape[0] // 2

        # Input X
        self.norm_train_x_seq = self.normal_seq[:n_seq_num, :, :]  # train
        self.norm_val_x_seq = self.normal_seq[n_seq_num:2 * n_seq_num, :, :]  # hyperopt
        self.norm_mahal_x_seq = self.normal_seq[2 * n_seq_num:3 * n_seq_num, :, :]  # mu and sigma
        self.norm_thres_x_seq = self.normal_seq[3 * n_seq_num:, :, :]  # threshold

        self.anom_thres_x_seq = self.anomaly_seq[:a_seq_num, :, :]  # threshold
        self.anom_val_x_seq = self.anomaly_seq[a_seq_num:, :, :]  # testing

        # Label
        self.norm_train_y_seq = self.normal_y_seq[:n_seq_num, :]
        self.norm_val_y_seq = self.normal_y_seq[n_seq_num:2 * n_seq_num, :]   # used in eval
        self.norm_mahal_y_seq = self.normal_y_seq[2 * n_seq_num:3 * n_seq_num, :]
        self.norm_thres_y_seq = self.normal_y_seq[3 * n_seq_num, :]

        self.anom_thres_y_seq = self.anomaly_y_seq[:a_seq_num, :]
        self.anom_val_y_seq = self.anomaly_y_seq[a_seq_num:, :]   # used in eval

    def _tuning(self, space):
        """Sets training parameters"""

        def objective(hyperparams):
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(self.norm_val_x_seq.shape[1], self.norm_val_x_seq.shape[2])),
                    keras.layers.LSTM(units=hyperparams['units']),
                    keras.layers.Dropout(rate=self.dropout_rate[0]),
                    keras.layers.RepeatVector(n=self.norm_val_x_seq.shape[1]),
                    keras.layers.LSTM(units=hyperparams['units'], return_sequences=True),
                    keras.layers.Dropout(rate=self.dropout_rate[1]),
                    keras.layers.TimeDistributed(keras.layers.Dense(self.norm_val_x_seq.shape[2]))
                ]
            )

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
                          loss=self.loss)

            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=self.config.train.early_stopping_rounds)

            model_history = model.fit(self.norm_val_x_seq,
                                      self.norm_val_x_seq,
                                      epochs=hyperparams['epochs'],
                                      batch_size=hyperparams['batch_size'],
                                      validation_split=self.val_subsplits,
                                      shuffle=False,
                                      callbacks=[es],
                                      use_multiprocessing=True
                                      )

            # Get the lowest validation loss of the training epochs
            validation_loss = np.amin(model_history.history['val_loss'])
            print('Best validation loss of epoch:', validation_loss)

            return {'loss': validation_loss,
                    'status': STATUS_OK,
                    'model': model,
                    'params': hyperparams,
                    'model_history': model_history}

        self.trials = Trials()

        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=self.max_evals)

        best_model = self.trials.results[np.argmin([r['loss'] for r in
                                                    self.trials.results])]['model']
        best_params = self.trials.results[np.argmin([r['loss'] for r in
                                                     self.trials.results])]['params']
        best_model_history = self.trials.results[np.argmin([r['loss'] for r in
                                                            self.trials.results])]['model_history']
        print(f"Optimized hyperparams: {best_params}")
        return best_params, best_model, best_model_history

    # ToDo: If Tune, then hyperparams to modify the respective attributes, and then build/train to use those but with the norm_train dataset

    def build(self):
        """ Builds the Keras model based """
        if self.tuning:
            self.hyperparams, self.model, self.model_history = self._tuning(model_spaces['lstm-ae'])
        else:
            self.model = keras.Sequential(
                [
                    keras.layers.Input(shape=(self.norm_train_x_seq.shape[1], self.norm_train_x_seq.shape[2])),
                    keras.layers.LSTM(units=self.units[0]),
                    keras.layers.Dropout(rate=self.dropout_rate[0]),
                    keras.layers.RepeatVector(n=self.norm_train_x_seq.shape[1]),
                    keras.layers.LSTM(units=self.units[1], return_sequences=True),
                    keras.layers.Dropout(rate=self.dropout_rate[1]),
                    keras.layers.TimeDistributed(keras.layers.Dense(self.norm_train_x_seq.shape[2]))
                ]
            )
        self.model.summary()

    def train(self):
        """Compiles and trains the model"""
        self.training_runtime = None
        if not self.tuning:
            print("Training the model ...")
            training_start_time = time.time()

            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config.train.learning_rate),
                               loss=self.loss)

            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=self.config.train.early_stopping_rounds)

            self.model_history = self.model.fit(self.norm_train_x_seq,
                                                self.norm_train_x_seq,
                                                epochs=self.epochs,
                                                batch_size=self.batch_size,
                                                validation_split=self.val_subsplits,
                                                shuffle=False,
                                                callbacks=[es],
                                                use_multiprocessing=True
                                                )

            self.stopped_iteration = es.stopped_epoch
            self.training_runtime = time.time() - training_start_time

    def eval(self):
        """Predicts results for the test dataset"""
        # Get the mahalanobis params (mean and cov) from the reconstruction error of a normal subset
        self.cov, self.mean = compute_mahalanobis_params(self.model, self.norm_mahal_x_seq)

        # Get the threshold from anomaly scores of a normal and anomaly subset
        normal_scores_thres = anomaly_scoring(self.model, self.norm_thres_x_seq, self.seq_time_steps,
                                               self.cov, self.mean)
        anomaly_scores_thres = anomaly_scoring(self.model, self.anom_thres_x_seq, self.seq_time_steps,
                                                self.cov, self.mean)
        self.threshold = compute_threshold(normal_scores_thres, anomaly_scores_thres)

        # Test normal subset
        n_accuracy, n_precision, n_recall, n_f1 = get_eval_metrics(self.model,
                                                                   self.norm_val_x_seq,
                                                                   self.norm_val_y_seq,
                                                                   self.threshold,
                                                                   self.seq_time_steps,
                                                                   self.cov, self.mean)

        # Log metrics to mlflow
        log_mlflow_metrics(n_accuracy, n_precision, n_recall, n_f1, 'train')

        # Test anomaly subset
        a_accuracy, a_precision, a_recall, a_f1 = get_eval_metrics(self.model,
                                                                   self.anom_val_x_seq,
                                                                   self.anom_val_y_seq,
                                                                   self.threshold,
                                                                   self.seq_time_steps,
                                                                   self.cov, self.mean)

        # Log metrics to mlflow
        log_mlflow_metrics(a_accuracy, a_precision, a_recall, a_f1, 'val')

    def _save_transformer(self):
        # Save time steps for creating sequences in transformer
        self.transformer.seq_time_steps = self.seq_time_steps
        # Save Mahalanobis params from train
        self.transformer.mahalanobis_params = (
            {"mean": self.mean, "cov": self.cov} if self.scores == "mahalanobis" else None)
        self.transformer.threshold = self.threshold
        # # Save mae_loss threshold
        # if self.scores == "mae_loss":
        #     self.transformer.mae_loss_threshold = compute_mae_loss_threshold(self.train_reconstruction_error)
        # Save the transformer
        self.transformer_name = self.datestr + "_transformer.sav"
        self.transformer_path = os.path.join(self.model_storage, self.transformer_name)
        joblib.dump(self.transformer, self.transformer_path)

    def save_model(self):
        """ Save the model and its artifacts. """
        self._save_transformer()
        if self.with_mlflow:
            """Save a serialized model"""
            print("Logging the model to MLflow ...")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                signature = infer_signature(self.norm_train_x_seq, self.model.predict(self.norm_train_x_seq))
                # artifact_path="model" is the location experiment>run>model
                mlflow.keras.log_model(self.model, artifact_path="model",
                                       signature=signature)
            else:
                mlflow.keras.log_model(self.model, "model")

            self._write_logs()

        # else:
        print('Saving the model and its artifacts ...')
        model_name = self.datestr + "_lstmae"
        keras.models.save_model(self.model, os.path.join(self.model_storage, model_name))

        return self.model_history.history['loss'], self.model_history.history['val_loss']

    def _write_logs(self):
        """Log training results"""

        # ToDo: same log for tune or without tune. the hyperparams are modified directly in the respective class attributes
        # ToDo: log tn, fp, fn, tp as metrics
        # Model hyper-parameters
        mlflow.log_params({
            "val_subsplits": self.val_subsplits,
            "train_shape": self.norm_train_x_seq.shape,
            "seq_time_steps": self.seq_time_steps,
            "dropout_rate": self.dropout_rate,
            "optimizer": self.model.optimizer._name,
            "criterion": self.loss,
            "anomaly_scores": self.scores
        })
        if self.tuning:
            mlflow.log_params({"num_iterations": self.hyperparams['epochs'],
                               "batch_size": self.hyperparams['batch_size'],
                               "units": self.hyperparams['units'],
                               "learning_rate": self.hyperparams['learning_rate']
                               })
        else:
            mlflow.log_params({"num_iterations": self.epochs,
                               "batch_size": self.batch_size,
                               "units": self.units,
                               "learning_rate": self.config.train.learning_rate,
                               "early_stopping_rounds": self.config.train.early_stopping_rounds,
                               "stopped_iteration": self.stopped_iteration
                               })

        # Training configuration file
        mlflow.log_artifact("modules/cybersecurity/src/configs/config.py")

        # Transformer
        mlflow.log_artifact(self.transformer_path)
        mlflow.set_tag("transformer_path", self.transformer_path)

        # Dataset name
        train_dataset_name = self.config.data.path_normal.split("/")[-1]
        mlflow.set_tag("train_dataset", train_dataset_name)
        test_dataset_name = self.config.data.path_anomaly.split("/")[-1]
        mlflow.set_tag("test_dataset", test_dataset_name)

        # tuning
        mlflow.set_tag("Tuning", self.tuning)

        # Model name
        mlflow.set_tag("model_name", self.config.model.model_name)

        # threshold
        mlflow.log_metric("threshold", self.threshold)
        mlflow.log_metric("loss", self.model_history.history["loss"][-1])
        mlflow.log_metric("val_loss", self.model_history.history["val_loss"][-1])

        # Training runtime
        if self.training_runtime:
            training_runtime_minutes = round(self.training_runtime / 60, 2)
            mlflow.log_metric("training_runtime", training_runtime_minutes)
