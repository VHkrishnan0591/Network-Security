from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
class DataTransformationPipeline():
    def __init__(self,data_validation_artifact):
        self.data_validation_artifact = data_validation_artifact

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(self.data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.data_transformation()
        return data_transformation_artifact
