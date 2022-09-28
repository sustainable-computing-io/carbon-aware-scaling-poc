
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY, GaugeMetricFamily
from confluent_kafka import Consumer
import time
import os, ast



class CO2SignalCollector():
    def __init__(self):

        self.TOPIC = os.getenv('TOPIC', 'carbon-intensity-topic')
        self.BOOTSTRAP_SERVER = os.getenv('BOOTSTRAP_SERVER','my-cluster-kafka-0.my-cluster-kafka-brokers.kafka.svc:9092')
        token = os.getenv('AUTH_TOKEN','')
        self.HEADERS = {'auth-token': token}
        self.LIMIT = 30
    
    
    def collect(self):
        print("Collect")   
 
        c = Consumer({
            'bootstrap.servers': self.BOOTSTRAP_SERVER,
            'group.id': 'mygroup',
            'enable.auto.commit': True
            })
        
        c.subscribe([self.TOPIC])
        count =1
        while count < self.LIMIT:
            count = count +1
            msg = c.poll(1)
            if msg is not None:
                print('Received message: {}'.format(msg.value().decode('utf-8')))
                
                message = msg.value()
                message=ast.literal_eval(message.decode('utf-8'))

                gauge = GaugeMetricFamily("carbon_intensity","Number to indicate carbon intensity",labels=[message['countryCode'],message['data']['datetime']])
                gauge.add_metric(['carbon_intensity'], message['data']['carbonIntensity'])
                print("Metric added")
                yield gauge 
        c.close()
            
      
    def export(self):
        print("export")
        start_http_server(9000)
        REGISTRY.register(CO2SignalCollector())

        while True:
            time.sleep(1)
            
                
        

if __name__ == "__main__":
    obj = CO2SignalCollector()
    obj.export()