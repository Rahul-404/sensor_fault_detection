import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from sensor_fault_detection.constant.training_pipeline import TARGET_COLUMN
from sensor_fault_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact)

from sensor_fault_detection.entity.config_entity import ModelEvaluationConfig
from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging
from sensor_fault_detection.ml.s3_estimator import SensorFaultEstimator


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        """
        :param model_evaluation_config: Output reference of data evaluation artifact stage
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param model_trainer_artifact: Output reference of model_trainer_artifact stage
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorFaultException(e, sys)
    
    def get_best_model(self) -> Optional[SensorFaultEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production

        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            heart_stroke_estimator = SensorFaultEstimator(
                bucket_name=bucket_name, model_path=model_path
            )

            if heart_stroke_estimator.is_model_present(model_path=model_path):
                return heart_stroke_estimator
            return None
        except Exception as e:
            raise HeartStrokeException(e, sys)
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model
                        with production model and choose best model

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            trained_model_f1_score = (
                self.model_trainer_artifact.metric_artifact.f1_score
            )

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)

            # calucate how much percentage training model accuracy is increased/decreased
            tmp_best_model_score = (
                0 if best_model_f1_score is None else best_model_f1_score
            )
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score,
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise SensorFaultException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise SensorFaultException(e, sys)