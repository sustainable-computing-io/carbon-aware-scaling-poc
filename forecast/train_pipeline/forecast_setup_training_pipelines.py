from abc import ABC, abstractmethod
from pandas import DataFrame


class GenericPreprocessingAndTrainingPipeline():

    def __init__(self) -> None:
        self.training_df = None
        self.

    @abstractmethod
    def generate_and_train_new_models():
        pass

    
    @abstractmethod
    def add_new_training_data_and_retrain():
        pass

    
    @abstractmethod
    def save_models():
        pass
    
    @abstractmethod
    def load_models_if_existing():
        pass

    @abstractmethod
    def predict_given_model_type_and_region():
        pass
