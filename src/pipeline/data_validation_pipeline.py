from src.config.configuration import ConfigurationManager
from src.components.data_validataion import DataValidation
class DataValidationPipeline():
    def __init__(self,data_ingestion_artifact):
        self.data_ingestion_artifact = data_ingestion_artifact

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config,self.data_ingestion_artifact)
        data_validation_artifact = data_validation.validating_data()
        return data_validation_artifact
