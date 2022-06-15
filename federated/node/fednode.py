import os

from .kafka_handler import KafkaAggregationProducer, KafkaAggregationConsumer

class FederatedNode(object):

    def __init__(self):
        self._mode = os.getenv('FED_BACKEND', 'kafka')
        if self._mode == 'kafka':
            self._params = {
                'broker_addr': f"{os.environ['KAFKA_HOST']}:{os.environ['KAFKA_PORT']}",
                'groupid': os.environ['GROUPID']
            }

            it = os.environ['TOPICS']
            self._topics = it.split(',') if ',' in it else [it]

        self._producer = None
        self._consumer = None

        self._build()
    

    def _build(self):
        print("Building the FederatedNode...")

        if self._mode == 'kafka':
            self._producer = KafkaAggregationProducer(self._params)
            self._consumer = KafkaAggregationConsumer(self._params, self._topics)
        else:
            raise NotImplementedError("Alternative communication protocols need implementation.")
        print("Done!")


    def __call__(self, service_fn):
        
        def service_pipeline(*args):
            obj = args[0]
            self._producer(service_fn(obj, self._consumer()))
        
        return service_pipeline