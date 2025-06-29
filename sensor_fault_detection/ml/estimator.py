import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging

class SensorFaultModel:

    def __init__(
            self,
            preprocessing_object: ColumnTransformer,
            trained_model_object: object
    ):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """ 
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last if performs prediction on transformed features.
        """
        logging.info("Entered predict method of HeartStrokeModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise SensorFaultException(e, sys)
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}"