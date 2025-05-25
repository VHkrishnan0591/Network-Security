from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
class DataIngestionPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.train_and_test_split()
        return data_ingestion_artifact
