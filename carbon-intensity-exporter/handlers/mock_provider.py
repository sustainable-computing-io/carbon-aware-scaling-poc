import time, os, json
from os.path import exists
import pandas as pd
from data_provider import DataProvider

nationalgrideso_file = 'df_fuel_ckan.csv' 
nationalgrideso_url = 'https://data.nationalgrideso.com/backend/dataset/88313ae5-94e4-4ddc-a790-593554d8c6b9/resource/f93d1835-75bc-43e5-84ad-12472b180a98/download/df_fuel_ckan.csv'

class MockProvider(DataProvider):
    def __init__(self, storage):
        self.pause_interval = int(os.getenv('PAUSE_INTERVAL', '5'))
        self.storage = storage
        self.data = list()
        test_file = nationalgrideso_file
        if exists(test_file) is False:
            test_file = nationalgrideso_url
        data = pd.read_csv(test_file, delimiter=',')

        ZONE = os.getenv('ZONE','GB')

        for index, d in data.iterrows():
            di = {}
            di['data'] = {}
            di['data']['carbonIntensity'] = d['CARBON_INTENSITY']
            di['data']['datetime'] = d['DATETIME']
            di['data']['fossilFuelPercentage'] = d['FOSSIL_perc']
            di['countryCode'] = ZONE

            self.data.append(json.dumps(di))
            


    def read(self):
        print("Read data")
        for d in self.data:
            co2_emissions_data= d
            self.storage.produce_one(co2_emissions_data)
            time.sleep(self.pause_interval)
