from GenericCarbonIntensityModel import GenericCarbonIntensityModel
import pandas as pd
import xgboost 
from neuralprophet import NeuralProphet


class NeuralProphetModel(GenericCarbonIntensityModel):
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, unique_model_id: str):
        super().__init__(co2_time_series_dataframe, unique_model_id)
        self.model = NeuralProphet(n_lags=3*24, n_forecasts=3*24, changepoints_range=0.90, n_changepoints=40, batch_size=50)
        self.train_df = None


    def preprocess_co2_time_series_df(self) -> None:
        if self.co2_time_series_dataframe.empty:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['ds', 'y'])
        clean_df['ds'] = pd.DatetimeIndex(self.co2_time_series_dataframe['datetime'])
        clean_df['y'] = pd.DatetimeIndex(self.co2_time_series_dataframe['carbon_intensity'])
        self.train_df = clean_df
        

    def train(self) -> None:
        if self.train_df is None:
            raise Exception("no trained dataframe")
        self.model.fit(df=self.train_df, freq="60min")


    def save_current_model(self) -> str:
        pass

    
    def load_existing_model(self, filepath: str) -> None:
        pass


    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        future = self.model.make_future_dataframe(self.train_df,periods=steps_into_future, n_historic_predictions=True)
        forecast = self.model.predict(future)
        self.model.plot(forecast)
        return forecast.tail(steps_into_future)


class XGBoostRegressorModel(GenericCarbonIntensityModel):
    def __init__(self, co2_time_series_dataframe: pd.DataFrame, region: Region):
        super().__init__(co2_time_series_dataframe, region)


    def preprocess_co2_time_series_df(self) -> None:
        return super().preprocess_co2_time_series_df()


    def train(self) -> None:
        return super().train()


    def save_current_model(self) -> str:
        return super().save_current_model()


    def load_existing_model(self, filepath: str):
        return super().load_existing_model(filepath)


    def forecast(self, steps_into_future: str) -> pd.DataFrame:
        return super().forecast(steps_into_future)
    