from .fedavg_aggregator import FedAvgAggregator


def get_aggregator(name: str):
    if name == 'fedavg':
        return FedAvgAggregator()
    else:
        raise ValueError(f"Parameter {name} is invalid.")