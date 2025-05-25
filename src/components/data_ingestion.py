from src.entity.config_entity import  DataIngestionArtifact
from src.utils.common import create_directories
from src.logging.logger import logger
from pymongo import MongoClient
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

class DataIngestion():
    def __init__(self,data_ingestion_config):
        self.data_ingestion_config = data_ingestion_config

    def data_ingestion_from_mongodb(self):

        # Creating the necessary directories
        logger.info("Creating the data ingestion directory")
        create_directories(self.data_ingestion_config.data_ingestion_directory)
        logger.info("Creating the raw data directory")
        create_directories(self.data_ingestion_config.raw_data_folder)
        logger.info("Creating the split data directory")
        create_directories(self.data_ingestion_config.split_data_folder)

        # Connecting to Mongo DB and ingesting data froom it
        logger.info("Getting the data from MongoDb")
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_DB_URL")  # or MongoDB Atlas URI

        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Access your database and collection
        db = client[self.data_ingestion_config.database_name]
        collection = db[self.data_ingestion_config.collection_name]

        # Fetch all documents
        cursor = collection.find()

        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))

        # Optional: remove the MongoDB internal _id field
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        df.replace({"na":np.nan},inplace=True)

        # Convert the dataframe into a raw_data.csv file
        df.to_csv(self.data_ingestion_config.raw_data_file, index=False)
        logger.info("Raw data is stored in the respective raw data folder")
        return df
    
    def train_and_test_split(self):
        logger.info("Raw data is split into train and test data")

        # Data Ingestion from mongoDB
        dataframe = self.data_ingestion_from_mongodb()
        X_train, X_test= train_test_split(
            dataframe, test_size=self.data_ingestion_config.test_train_split, random_state=42)
        
        # Convert the dataframe into a train_data.csv file
        X_train.to_csv(self.data_ingestion_config.train_filepath, index=False)

        # Convert the dataframe into a test_data.csv file
        X_test.to_csv(self.data_ingestion_config.test_filepath, index=False)

        logger.info("train and test data is stored in ingested folder")

        return DataIngestionArtifact(
            train_filepath = self.data_ingestion_config.train_filepath,
            test_filepath = self.data_ingestion_config.test_filepath
        )
