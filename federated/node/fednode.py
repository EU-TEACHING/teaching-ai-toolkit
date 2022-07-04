import os


class FederatedNode(object):

    def __init__(self, produce=True, consume=True):
        self.activated = os.getenv('MODE') == 'FEDERATED'
        if self.activated:
            self._mode = os.getenv('FED_BACKEND', 'kafka')
            
            if self._mode == 'kafka':
                self._params = {
                    'broker_addr': f"{os.getenv('KAFKA_HOST')}:{os.getenv('KAFKA_PORT')}",
                    'groupid': os.getenv('GROUPID'),
                    'timeout': float(os.getenv('TIMEOUT', '0.5'))
                }

                it = os.getenv('FED_TOPICS')
                self._topics = it.split(',') if ',' in it else [it]

            elif self._mode == 'fs':
                self._params = {
                    'produce_dir': os.getenv('FS_PRODUCE_DIR'),
                    'consume_dir': os.getenv('FS_CONSUME_DIR')
                }
            self._produce, self._consume = produce, consume
            self._producer, self._consumer = None, None

            self._build()
    

    def _build(self):

        if self._mode == 'kafka':
            print("Building the Kafka FederatedNode...")
            from .communication.kafka_handler import KafkaAggregationProducer, KafkaAggregationConsumer
            if self._produce:
                self._producer = KafkaAggregationProducer(self._params)
            if self._consume:
                self._consumer = KafkaAggregationConsumer(self._params)

        elif self._mode == 'fs':
            print("Building the FS handler for the FederatedNode...")
            from .communication.fs_handler import FileSystemProducer, FileSystemConsumer
            if self._produce:
                self._producer = FileSystemProducer(self._params['produce_dir'])
            if self._consume:
                self._consumer = FileSystemConsumer(self._params['consume_dir'])

        else:
            raise NotImplementedError("Alternative communication protocols need implementation.")
        print("Done!")


    def __call__(self, service_fn):
        
        def service_pipeline(*args):
            obj = args[0]
            if self.activated and (self._consume or self._produce):
                if self._mode == 'kafka' and self._consume:
                    self._consumer.subscribe(obj.get_subscribe_topics())

                if self._consume and not self._produce:
                    service_fn(obj, self._consumer())


                if not self._consume and self._produce:
                    self._producer(service_fn(obj))


                if self._consume and self._produce:
                    self._producer(service_fn(obj, self._consumer()))
            else:
                service_fn(obj, args[1])
        
        return service_pipeline