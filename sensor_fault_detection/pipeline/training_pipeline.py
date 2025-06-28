
from sensor_fault_detection.components.data_ingestion import DataIngestion

from sensor_fault_detection.entity.config_entity import (DataIngestionConfig)

from sensor_fault_detection.entity.artifact_entity import (DataIngestionArtifact)

from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging
import sys

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise SensorFaultException(e, sys)
        

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
            logging.info(f"Training Pipeline is complete. {data_ingestion_artifact}")

        except Exception as e:
            raise SensorFaultException(e, sys)