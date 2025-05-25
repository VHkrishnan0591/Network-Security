from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining
class ModelTrainingPipeline():
    def __init__(self,data_transformation_artifact):
        self.data_transformation_artifact = data_transformation_artifact

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(self.data_transformation_artifact,model_training_config)
        model_training_artifact = model_training.model_training()
        return model_training_artifact