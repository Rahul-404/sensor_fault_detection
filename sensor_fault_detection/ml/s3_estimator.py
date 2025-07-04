import os
import sys

from pandas import DataFrame

from sensor_fault_detection.cloud_storage.aws_storage import SimpleStorageService
from sensor_fault_detection.ml.estimator import SensorFaultModel
from sensor_fault_detection.exception import SensorFaultException


class SensorFaultEstimator:
    """ Load from s3 and then make it usable for prediction"""
    def __init__(
        self,
        bucket_name,
        model_path,
    ):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: SensorFaultModel = None
    
    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name, s3_key=model_path
            )
        except SensorFaultException as e:
            print(e)
            return False
    
    def load_model(self,) -> SensorFaultModel:
        """
        Load the model from the model_path
        :return:
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
    
    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise SensorFaultException(e, sys)
    
    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise SensorFaultException(e, sys)