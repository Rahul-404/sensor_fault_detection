import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder

from sensor_fault_detection.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from sensor_fault_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sensor_fault_detection.entity.config_entity import DataTransformationConfig
from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging
from sensor_fault_detection.utils.main_utils import (
    read_yaml_file,
    save_numpy_array_data,
    save_object,
)

class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorFaultException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorFaultException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object

        Output      :   data transformer object is created and returned
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info(
                "Entered into get_data_transformer_object method of DataTransformation class"
            )

            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler = RobustScaler()

            pipeline = Pipeline(
                steps=[
                    ("imputer", simple_imputer),
                    ("scaler", robust_scaler),
                ]
            )

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return pipeline

        except Exception as e:
            raise SensorFaultException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline

        Output      :   data transformer steps are performed and preprocessor object is created
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Testing dataset")

                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info("Used the preprocessor object to fit transform the train features")

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)


                logging.info("Converting target categorical column into nummerical column for train and test")

                label_encoder = LabelEncoder()
                label_encoder.fit(target_feature_train_df)

                target_feature_train_df = label_encoder.transform(target_feature_train_df)
                target_feature_test_df = label_encoder.transform(target_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTETomek on Training dataset")

                smt = SMOTETomek(random_state=42, sampling_strategy='minority', n_jobs=-1)

                (
                    input_feature_train_final,
                    target_feature_train_final,
                ) = smt.fit_resample(input_feature_train_arr, target_feature_train_df)

                logging.info("Applied SMOTETomek on training dataset")

                logging.info("Applying SMOTETomek on testing dataset")

                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Applied SMOTETomek on testing dataset")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(
                    self.data_transformation_config.transformer_object_file_path,
                    preprocessor,
                )

                save_object(
                    self.data_transformation_config.label_encoder_object_file_path,
                    label_encoder,
                )

                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path,
                    array=train_arr,
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path,
                    array=test_arr,
                )

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformer_object_file_path=self.data_transformation_config.transformer_object_file_path,
                    label_encoder_object_file_path = self.data_transformation_config.label_encoder_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise SensorFaultException(e, sys)