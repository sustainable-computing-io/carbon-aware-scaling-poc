from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY, GaugeMetricFamily
import time
import os, ast, json

class CO2SignalCollector():
    def __init__(self, storage):
        self.storage = storage
    
    def collect(self):
        print("Collect")   
        while True:
            message = self.storage.consume_one()
            if message is not None:
                message = json.loads(message)
                print(message)
                gauge = GaugeMetricFamily("carbon_intensity_zone_" + message['countryCode'],
                    "Number to indicate carbon intensity in zone " + message['countryCode'], labels=['datetime'])
                # FIXME: adjust to timezone
                gauge.add_metric([message['data']['datetime']], message['data']['carbonIntensity'])
                print("Metric added")
                yield gauge 
            else:
                break


    def export(self, storage):
        print("export")
        start_http_server(9000)
        REGISTRY.register(CO2SignalCollector(storage))

        while True:
            time.sleep(1)
