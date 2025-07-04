import os.path
import sys
import pandas as pd
import dill
import numpy as np
import yaml
import json
from sensor_fault_detection.exception import SensorFaultException
from sensor_fault_detection.logger import logging
from sensor_fault_detection.configuration import mongo_db_connection


def dump_csv_file_to_mongodb_collection(file_path: str, database_name: str, collection_name: str) -> None:
    try:
        # reading the csv file
        df = pd.read_csv(file_path)
        logging.info(f"Rows and Columns {df.shape}")
        df.reset_index(drop=True, inplace=True)
        json_records = list(json.load(df.T.to_json()).values())
        mongo_db_connection.MongoDBClient[database_name][collection_name].insert_many(json_records)
    except Exception as e:
        raise SensorFaultException(e, sys)

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise SensorFaultException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise SensorFaultException(e, sys)

def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of MainUtils class")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of MainUtils class")

        return obj

    except Exception as e:
        raise SensorFaultException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorFaultException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorFaultException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of MainUtils class")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise SensorFaultException(e, sys) from e
