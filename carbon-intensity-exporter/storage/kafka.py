import confluent_kafka
import os
from .storage_provider import StorageProvider

class KafkaStorage(StorageProvider):

    def __init__(self):
        self.topic = os.getenv('TOPIC', 'carbon-intensity-topic')
        self.bootstrap_server = os.getenv('BOOTSTRAP_SERVER','my-cluster-kafka-0.my-cluster-kafka-brokers.kafka.svc:9092')

        self.producer = confluent_kafka.Producer({'bootstrap.servers': 'my-cluster-kafka-0.my-cluster-kafka-brokers.kafka.svc:9092'})
        self.consumer = confluent_kafka.Consumer({
            'bootstrap.servers': self.BOOTSTRAP_SERVER,
            'enable.auto.commit': True
            })

    def __kakfa_callback(err, msg):
        if err is not None:
            print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
        else:
            message = 'Produced message on topic {} with value of {}\n'.format(msg.topic(), msg.value().decode('utf-8'))
            print(message)

    def produce_one(self, data):
        self.producer.produce(self.topic, data, callback=__kakfa_callback)
        self.producer.poll(1)

    def consume_one(self):
        self.consumer.subscribe([self.topic])
        while True:
            msg = self.consumer.poll(timeout=1.0)
            if msg is not None:
                self.consumer.close()
                return msg        
        self.consumer.close()
