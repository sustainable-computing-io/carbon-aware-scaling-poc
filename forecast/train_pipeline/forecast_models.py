from xxlimited import foo
import pandas as pd
import xgboost 
from neuralprophet import NeuralProphet
from forecast_types import Region
from abc import ABC, abstractmethod
from forecast_types import Region, Step_Type, step_type_to_neural_prophet_freq, step_to_m
import os
from pmdarima import auto_arima
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class GenericCarbonIntensityForecastModel(ABC):

    # time series dataframe requires datetime and carbon intensity and region
    # precondition: carbon intensity time series data need to belong to a single region
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, step_type: Step_Type, save_location="default"):
        self.region = region
        self.co2_time_series_dataframe = co2_time_series_dataframe
        self.save_location = save_location
        self.step_type = step_type
        self.filename = "{}_{}.pkl".format(self.__class__.__name__, self.region.name)
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


    @abstractmethod
    def create_new_model(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def _preprocess_co2_time_series_df(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    def _save_co2_time_series_dataframe(self):
        try: 
            self.co2_time_series_dataframe.to_pickle(os.path.join(self.save_location, self.filename))
        except Exception:
            raise Exception("could not save dataframe")


    def _load_co2_time_series_dataframe(self):
        try:
            self.co2_time_series_dataframe = pd.read_pickle(os.path.join(self.save_location, self.filename))
        except Exception:
            raise Exception("could not load dataframe")

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
        super().__init__(co2_time_series_dataframe, region, step_type, save_location)
        self.filename = "{}_{}.pkl".format(self.__class__.__name__, self.region.name)
        self.train_df = None
        self.create_new_model()


    def create_new_model(self) -> None:
        self._preprocess_co2_time_series_df()
        self.model = NeuralProphet(n_lags=3*24, n_forecasts=3*24, changepoints_range=0.90, n_changepoints=30, batch_size=50)

    def append_new_co2_time_series_dataframe(self, df: pd.DataFrame) -> None:
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self.co2_time_series_dataframe = pd.concat([self.co2_time_series_dataframe, clean_df])
    

    def _preprocess_co2_time_series_df(self) -> None:
        if self.co2_time_series_dataframe.empty:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['ds', 'y'])
        clean_df['ds'] = pd.DatetimeIndex(self.co2_time_series_dataframe['datetime'])
        clean_df['y'] = self.co2_time_series_dataframe['carbon_intensity']
        clean_df.dropna()
        self.train_df = clean_df
        

    def train(self) -> None:
        if self.train_df is None:
            raise Exception("no trained dataframe")
        self.model.fit(df=self.train_df, freq=step_type_to_neural_prophet_freq[self.step_type])


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
        super().__init__(co2_time_series_dataframe, region, Step_Type.Half_Hour, save_location)
        self.filename = "{}_{}.pkl".format(self.__class__.__name__, self.region.name)
        self.train_df = None
        self.test_df = None
    

    def create_new_model(self) -> None:
        self._preprocess_co2_time_series_df()
        

    def _preprocess_co2_time_series_df(self) -> None:
        if self.co2_time_series_dataframe.empty:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['Date', 'value'])
        clean_df['Date'] = pd.to_datetime(self.co2_time_series_dataframe['datetime'])
        clean_df.dropna()
        clean_df.set_index(clean_df['Date'], inplace=True)
        clean_df['value'] = self.co2_time_series_dataframe['carbon_intensity']
        split_idx = round(len(clean_df)* 0.8)
        self.train_df = clean_df[:split_idx]
        self.test_df = clean_df[split_idx:]

    
    def __forecast_one_step(self):
        fc, conf = self.model.predict(n_periods=1, return_conf_int=True)
        return (fc.tolist()[0], np.asarray(conf).tolist()[0])

    
    def __forecast_test_dataset(self):
        forecasts = []
        confidence_intervals = []
        for t in self.test_df:
            fc, conf = self.__forecast_one_step()
            forecasts.append(fc)
            confidence_intervals.append(conf)
            self.model.update(t)
        
        return forecasts, confidence_intervals, mean_absolute_error(self.test_df, forecasts), r2_score(self.test_df, forecasts)

    
    def train(self) -> None:
        if self.model is None:
            self.model = auto_arima(self.train_df, seasonal=True, m=step_to_m[self.step_type])
            f, ci, mae, r2 = self.__forecast_test_dataset()
        else:
            self.model.update(self.train_df)
            f, ci, mae, r2 = self.__forecast_test_dataset()
        return f, ci, mae, r2


    def save(self) -> None:
        self._save_co2_time_series_dataframe()
        # save model
        try:
            joblib.dump(self.model, os.path.join(self.save_location, self.filename))
        except:
            raise Exception("failed to save model")


    def load(self) -> None:
        self._load_co2_time_series_dataframe()
        # reload model
        try:
            self.model = joblib.load(os.path.join(self.save_location, self.filename))
        except:
            raise Exception("failed to reload model")
    

    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        fc, confint = self.model.predict(n_periods=steps_into_future, return_conf_int=True)
        return fc, confint


class XGBoostRegressorModel(GenericCarbonIntensityForecastModel):

    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region, model_type: Model_Type, step_type: Step_Type, save_location="default"):
        super().__init__(co2_time_series_dataframe, region, model_type, step_type, save_location)

    
    def create_new_model(self) -> None:
        return super().create_new_model()


    def _preprocess_co2_time_series_df(self) -> None:
        return super()._preprocess_co2_time_series_df()

    
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