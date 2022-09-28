from abc import ABC, abstractmethod
import pandas as pd
from forecast_types import Model_Type

class GenericCarbonIntensityModel(ABC):

    def __init__(self, raw_train_dataframe: pd.DataFrame, unique_model_id: str, model_type: Model_Type):
        self._raw_train_dataframe = raw_train_dataframe
        self.model_type = model_type
        self.unique_model_id = unique_model_id
        self.model = None


    @abstractmethod
    def create_default_model(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def preprocess_input_dataframe(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def save_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, filepath: str):
        raise NotImplementedError 


    @abstractmethod
    def predict(self, future_dataframe: pd.DataFrame):
        raise NotImplementedError
