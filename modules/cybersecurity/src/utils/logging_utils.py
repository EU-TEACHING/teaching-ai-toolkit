import mlflow
from mlflow.tracking import MlflowClient
from modules.cybersecurity.mlflow_env_vars import mlflow_connect


def mlflow_config(mlflow_cfg):
    mlflow_connect()
    tags = mlflow_cfg["tags"]
    experiment_name = mlflow_cfg["experiment_name"]
    experiment_id = retrieve_mlflow_experiment_id(experiment_name, create=True)

    return mlflow.start_run(experiment_id=experiment_id, tags=tags)


def retrieve_mlflow_experiment_id(name, create=False):
    experiment_id = None
    if name:
        existing_experiment = MlflowClient().get_experiment_by_name(name)
        if existing_experiment and existing_experiment.lifecycle_stage == 'active':
            experiment_id = existing_experiment.experiment_id
        else:
            if create:
                experiment_id = mlflow.create_experiment(name)
            else:
                raise Exception(f'Experiment "{name}" not found in {mlflow.get_tracking_uri()}')

    if experiment_id is not None:
        experiment = MlflowClient().get_experiment(experiment_id)
        print("Experiment name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    return experiment_id


def log_mlflow_metrics(accuracy, precision, recall, f1, mode):
    mlflow.log_metric(f'Accuracy_{mode}', accuracy)
    mlflow.log_metric(f'Precision_{mode}', precision)
    mlflow.log_metric(f'Recall_{mode}', recall)
    mlflow.log_metric(f'F1_{mode}', f1)
