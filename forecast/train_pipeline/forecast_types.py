from enum import Enum


class Model_Type(Enum):
    XGBoosterRegressionModel = 1
    NeuralProphet = 2
    ARIMA = 3
    LSTM = 4
    LGBMRegressionModel = 5


model_type_to_features_labels = {
    Model_Type.XGBoosterMLRegressionModel: ((), ()),
    Model_Type.NeuralProphet: (("ds"), ("y")),
    Model_Type.ARIMA: ((), ()),
    Model_Type.LSTM: ((), ()),
    Model_Type.LGBMRegressionModel: ((), ())
    

}