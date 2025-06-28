import os
import sys

import certifi
import pymongo

from sensor_fault_detection.constant.database import DATABASE_NAME
from sensor_fault_detection.constant.env_variable import MONGODB_URL_KEY
from sensor_fault_detection.exception import SensorFaultException

ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe

    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """

    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise SensorFaultException(e, sys)


