from GenericCarbonIntensityModel import GenericCarbonIntensityModel
from forecast_types import Model_Type
from pandas import pd
import xgboost 
from neuralprophet import NeuralProphet


class NeuralProphetModel(GenericCarbonIntensityModel):
    def __init__(self, raw_train_dataframe: pd.DataFrame, unique_model_id: str, model_type: Model_Type):
        super().__init__(raw_train_dataframe, unique_model_id, model_type)

    def create_default_model(self) -> None:
        if self.model is None:
            self.model = NeuralProphet()
        

    def preprocess_input_dataframe(self) -> None:
        return super().preprocess_input_dataframe()


    def train(self) -> None:
        return super().train()

    def save_model(self) -> str:
        return super().save_model()

    
    def load_model(self, filepath: str):
        return super().load_model(filepath)

    def predict(self, future_dataframe: pd.DataFrame):
        return super().predict(future_dataframe)


class XGBoostRegressorModel(GenericCarbonIntensityModel):
    def __init__(self, raw_train_dataframe: pd.DataFrame, unique_model_id: str, model_type: Model_Type):
        super().__init__(raw_train_dataframe, unique_model_id, model_type)

    def create_default_model(self) -> None:
        if self.model is None:
            return super().create_default_model()


    def preprocess_input_dataframe(self) -> None:
        return super().preprocess_input_dataframe()


    def train(self) -> None:
        return super().train()

    def save_model(self) -> str:
        return super().save_model()

    
    def load_model(self, filepath: str):
        return super().load_model(filepath)

    
    def predict(self, future_dataframe: pd.DataFrame):
        return super().predict(future_dataframe)