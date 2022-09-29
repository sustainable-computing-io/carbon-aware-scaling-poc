from abc import ABC, abstractmethod
from matplotlib.pyplot import step
import pandas as pd
from forecast_types import Region, Step_Type


class GenericCarbonIntensityForecastModel(ABC):

    # time series dataframe requires datetime and carbon intensity and region
    # precondition: carbon intensity time series data need to belong to a single region
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, step_type: Step_Type, save_location="default"):
        self.region = region
        self.co2_time_series_dataframe = co2_time_series_dataframe
        self.save_location = save_location
        self.step_type = step_type
        self.num_of_missing_carbon_intensity_values = 0


    def _validate_and_clean_co2_time_series_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df and "carbon_intensity" in df and "region" in df:
            if (df['region'] == self.region).all():
                clean_df = pd.DataFrame(columns=['datetime', 'carbon_intensity', 'region'])
                clean_df['datetime'] = df['datetime']
                clean_df['carbon_intensity'] = df['carbon_intensity']
                clean_df['region'] = df['region']
                
                return clean_df
            raise Exception("carbon intensity time series data do not belong to a single region")
        raise Exception("missing required columns: datetime, carbon_intensity, region")
    

    @property
    def co2_time_series_dataframe(self):
        return self.__co2_time_series_dataframe


    @co2_time_series_dataframe.setter
    def co2_time_series_dataframe(self, df: pd.DataFrame):
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self.__co2_time_series_dataframe = clean_df

            
    def append_new_co2_time_series_dataframe(self, df: pd.DataFrame):
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self.co2_time_series_dataframe = pd.concat([self.co2_time_series_dataframe, clean_df])
    

    @abstractmethod
    def preprocess_co2_time_series_df(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def _save_co2_time_series_dataframe():
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
