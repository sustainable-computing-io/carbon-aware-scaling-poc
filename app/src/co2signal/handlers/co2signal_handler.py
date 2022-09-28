import confluent_kafka

import time
import requests, json,  itertools, os

class CO2SignalHandler():

    def __init__(self):

        self.TOPIC = os.getenv('TOPIC', 'carbon-intensity-topic')
        self.BOOTSTRAP_SERVER = os.getenv('BOOTSTRAP_SERVER','my-cluster-kafka-0.my-cluster-kafka-brokers.kafka.svc:9092')

        self.EMISSION_URL = os.getenv('EMIISION_URL','https://api.co2signal.com/v1/latest')
        token = os.getenv('AUTH_TOKEN','')
        self.HEADERS = {'auth-token': token}
      
        self.LIMIT = 30

    def __chunked(self,it, size):
            it = iter(it)
            while True:
                p = dict(itertools.islice(it, size))
                if not p:
                    break
                yield p
    
    def __getCountryCodes(self):
        country_codes_url = 'http://api.electricitymap.org/v3/zones'
        country_codes = requests.get(url=country_codes_url).json()
        return country_codes


    def read(self):
        print("Read data")

        def callback(err, msg):
            if err is not None:
                print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
            else:
                message = 'Produced message on topic {} with value of {}\n'.format(msg.topic(), msg.value().decode('utf-8'))
                print(message)

        emissions_producer = confluent_kafka.Producer({'bootstrap.servers': 'my-cluster-kafka-0.my-cluster-kafka-brokers.kafka.svc:9092'})
        # parameter = {'countryCode': '{}'.format("DK")}
        # co2_emissions_data= requests.get(url=self.EMISSION_URL, headers=self.HEADERS, params=parameter).json()


        for chunk in self.__chunked(self.__getCountryCodes().items(), self.LIMIT):
            for key,value in chunk.items():
                parameter = {'countryCode': '{}'.format(key)}
                co2_emissions_data= requests.get(url=self.EMISSION_URL, headers=self.HEADERS, params=parameter).json()
                if 'data' not in co2_emissions_data:
                    continue
                data = co2_emissions_data['data']
                if data['fossilFuelPercentage'] is not None:
                    emissions_producer.produce(self.topic, json.dumps(co2_emissions_data).encode('utf-8'), callback=callback)
                    emissions_producer.poll(1)
            time.sleep(3600)


if __name__ == "__main__":
    obj = CO2SignalHandler()
    obj.read()