import os

from .kafka_handler import KafkaAggregationProducer, KafkaAggregationConsumer

class FederatedNode(object):

    def __init__(self, produce=True, consume=True):
        self._mode = os.getenv('FED_COMM_BACKEND', 'kafka')
        if self._mode == 'kafka':
            self._params = {
                'broker_addr': f"{os.environ['KAFKA_HOST']}:{os.environ['KAFKA_PORT']}",
                'groupid': os.environ['GROUPID']
            }

            it = os.environ['TOPICS']
            self._topics = it.split(',') if ',' in it else [it]

        self._produce, self._consume = produce, consume
        self._producer, self._consumer = None, None

        self._build()
    

    def _build(self):

        if self._mode == 'kafka':
            print("Building the Kafka FederatedNode...")
            if self._produce:
                self._producer = KafkaAggregationProducer(self._params)
            if self._consume:
                self._consumer = KafkaAggregationConsumer(self._params, self._topics)
        else:
            raise NotImplementedError("Alternative communication protocols need implementation.")
        print("Done!")


    def __call__(self, service_fn):
        
        def service_pipeline(*args):
            obj = args[0]
            if self._consume and not self._produce:
                service_fn(obj, self._consumer())


            if not self._consume and self._produce:
                self._producer(service_fn(obj))


            if self._consume and self._produce:
                self._producer(service_fn(obj, self._consumer()))
        
        return service_pipeline