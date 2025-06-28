import json
import sys

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pandas import DataFrame

from sensor_fault_detection.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor_fault_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)

from sensor_fault_detection.entity.config_entity import DataValidationConfig
from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging
from sensor_fault_detection.utils.main_utils import read_yaml_file, write_yaml_file
import os
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorFaultException(e, sys)
    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            return status
        except Exception as e:
            raise SensorFaultException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise SensorFaultException(e, sys)

    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
    ) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            data_drift_profile = Report([DataDriftPreset()])
            data_drift_profile.run(reference_data=reference_df, current_data=current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            n_features = json_report["metrics"][0]["result"]["number_of_columns"]
            n_drifted_features = json_report["metrics"][0]["result"]["number_of_drifted_columns"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")

            drift_status = json_report["metrics"][0]['result']['dataset_drift']
            return drift_status
        except Exception as e:
            raise SensorFaultException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                ),
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                ),
            )

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(
                f"Total number of required columns present in training dataframe: {status}"
            )

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(
                f"Total number of required columns present in testing dataframe: {status}"
            )

            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."


            validation_status = len(validation_error_msg) == 0
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Data Drift detected.")
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorFaultException(e, sys)