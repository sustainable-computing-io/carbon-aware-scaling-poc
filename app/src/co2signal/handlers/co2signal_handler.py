import time
import requests, json,  itertools, os
from .data_provider import DataProvider

class CO2SignalHandler(DataProvider):

    def __getCountryCodes(self):
        country_codes_url = 'http://api.electricitymap.org/v3/zones'
        country_codes = requests.get(url=country_codes_url).json()
        return country_codes

    def __init__(self, storage):
        self.EMISSION_URL = os.getenv('EMIISION_URL','https://api.co2signal.com/v1/latest')
        token = os.getenv('AUTH_TOKEN','')
        self.HEADERS = {'auth-token': token}
        self.storage = storage
        self.country_codes = self.__getCountryCodes()
        self.pause_interval = int(os.getenv('PAUSE_INTERVAL', '120'))

    def read(self):
        print("Read data")

        while True:
            for key in self.country_codes.keys():
                parameter = {'countryCode': '{}'.format(key)}
                try:
                    co2_emissions_data= requests.get(url=self.EMISSION_URL, headers=self.HEADERS, params=parameter).json()
                    print(co2_emissions_data)
                    if 'data' not in co2_emissions_data:
                        continue
                    data = co2_emissions_data['data']
                    if data['fossilFuelPercentage'] is not None:
                        self.storage.produce_one(json.dumps(co2_emissions_data).encode('utf-8'))
                except:
                    print('failed to query')
                    pass

                time.sleep(self.pause_interval)
