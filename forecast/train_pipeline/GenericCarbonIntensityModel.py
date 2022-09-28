from abc import ABC, abstractmethod
import pandas as pd
from forecast_types import Region


class GenericCarbonIntensityModel(ABC):

    # time series dataframe requires datetime and carbon intensity
    # precondition: carbon intensity time series data belong to a single region
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region):
        self.region = region
        self.co2_time_series_dataframe = co2_time_series_dataframe


    @property
    def co2_time_series_dataframe(self):
        return self.__co2_time_series_dataframe


    @co2_time_series_dataframe.setter
    def co2_time_series_dataframe(self, df: pd.DataFrame):
        # include cleaning logic
        if "datetime" in df and "carbon_intensity" in df:
            clean_df = df.dropna()
            self.__co2_time_series_dataframe = clean_df
    

    @abstractmethod
    def preprocess_co2_time_series_df(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def save_current_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_existing_model(self, filepath: str):
        raise NotImplementedError 


    @abstractmethod
    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        raise NotImplementedError
