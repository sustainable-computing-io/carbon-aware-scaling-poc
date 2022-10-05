from xxlimited import foo
import pandas as pd
import xgboost 
from neuralprophet import NeuralProphet
from forecast_types import Region
from abc import ABC, abstractmethod
from forecast_types import Region, Step_Type, Model_Type, step_type_to_neural_prophet_freq
import os
from pmdarima import auto_arima


class GenericCarbonIntensityForecastModel(ABC):

    # time series dataframe requires datetime and carbon intensity and region
    # precondition: carbon intensity time series data need to belong to a single region
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, model_type: Model_Type, step_type: Step_Type, save_location="default"):
        self.region = region
        self.co2_time_series_dataframe = co2_time_series_dataframe
        self.save_location = save_location
        self.step_type = step_type
        self.model_type = model_type
        self._filename = "{}_{}".format(self.model_type.name, self.region.name)
        # Can I assume that the time series dataframe is complete? (or at the very least there are no missing time points?)
        #self.num_of_missing_carbon_intensity_values = 0
        self.model = None


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
    def save_location(self) -> str:
        return self.__save_location


    @save_location.setter
    def save_location(self, folderpath: str):
        if folderpath == "default":
            dirname=os.path.dirname
            self.__save_location = os.path.join(os.path.dirname(dirname(__file__)), "models")
        else:
            if os.path.exists(folderpath):
                self.__save_location = folderpath
            

    @property
    def co2_time_series_dataframe(self) -> pd.DataFrame:
        return self.__co2_time_series_dataframe


    @co2_time_series_dataframe.setter
    def co2_time_series_dataframe(self, df: pd.DataFrame):
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self.__co2_time_series_dataframe = clean_df


    def append_new_co2_time_series_dataframe(self, df: pd.DataFrame) -> None:
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self.co2_time_series_dataframe = pd.concat([self.co2_time_series_dataframe, clean_df])
    

    @abstractmethod
    def create_new_model(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def preprocess_co2_time_series_df(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def _save_co2_time_series_dataframe(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _load_co2_time_series_dataframe(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError 


    @abstractmethod
    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        raise NotImplementedError



class NeuralProphetModel(GenericCarbonIntensityForecastModel):
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, step_type=Step_Type.Auto, save_location="default"):
        super().__init__(co2_time_series_dataframe, region, Model_Type.NeuralProphetModel, step_type, save_location)
        self.train_df = None
        self.create_new_model()


    def create_new_model(self) -> None:
        self.model = NeuralProphet(n_lags=3*24, n_forecasts=3*24, changepoints_range=0.90, n_changepoints=30, batch_size=50)


    def preprocess_co2_time_series_df(self) -> None:
        if self.co2_time_series_dataframe.empty:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['ds', 'y'])
        clean_df['ds'] = pd.DatetimeIndex(self.co2_time_series_dataframe['datetime'])
        clean_df['y'] = self.co2_time_series_dataframe['carbon_intensity']
        self.train_df = clean_df
        

    def train(self) -> None:
        if self.train_df is None:
            raise Exception("no trained dataframe")
        self.model.fit(df=self.train_df, freq=step_type_to_neural_prophet_freq[self.step_type])


    def _save_co2_time_series_dataframe(self):
        try: 
            self.co2_time_series_dataframe.to_pickle(os.path.join(self.save_location, self._filename))
        except Exception:
            raise Exception("could not save dataframe")


    def _load_co2_time_series_dataframe(self):
        try:
            self.co2_time_series_dataframe = pd.read_pickle(os.path.join(self.save_location, self._filename))
        except Exception:
            raise Exception("could not load dataframe")


    def save(self):
        self._save_co2_time_series_dataframe()


    def load(self):
        self._load_co2_time_series_dataframe()


    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        future = self.model.make_future_dataframe(self.train_df,periods=steps_into_future, n_historic_predictions=True)
        forecast = self.model.predict(future)
        self.model.plot(forecast)
        return forecast.tail(steps_into_future)



class ARIMAModel(GenericCarbonIntensityForecastModel):
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, save_location="default"):
        super().__init__(co2_time_series_dataframe, region, Model_Type.ARIMA, Step_Type.Auto, save_location)
    

    def create_new_model(self) -> None:
        self.model = auto_arima()
    

    def preprocess_co2_time_series_df(self) -> None:
        return super().preprocess_co2_time_series_df()

    
    def train(self) -> None:
        return super().train()

    
    def _save_co2_time_series_dataframe(self) -> None:
        return super()._save_co2_time_series_dataframe()


    def _load_co2_time_series_dataframe(self) -> None:
        return super()._load_co2_time_series_dataframe()


    def save(self) -> None:
        return super().save()

    
    def load(self) -> None:
        return super().load()

    
    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        return super().forecast(steps_into_future)


class XGBoostRegressorModel(GenericCarbonIntensityForecastModel):

    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, model_type: Model_Type, step_type: Step_Type, save_location="default"):
        super().__init__(co2_time_series_dataframe, region, model_type, step_type, save_location)

    
    def create_new_model(self) -> None:
        return super().create_new_model()


    def preprocess_co2_time_series_df(self) -> None:
        return super().preprocess_co2_time_series_df()

    
    def train(self) -> None:
        return super().train()

    
    def _save_co2_time_series_dataframe(self) -> None:
        return super()._save_co2_time_series_dataframe()


    def _load_co2_time_series_dataframe(self) -> None:
        return super()._load_co2_time_series_dataframe()


    def save(self) -> None:
        return super().save()

    
    def load(self) -> None:
        return super().load()

    
    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        return super().forecast(steps_into_future)