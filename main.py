from src.logging.logger import logger
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.entity.config_entity import DataTransformationArtifact
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        data_ingestion_artifact = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = 'Data Validation Stage'
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline(data_ingestion_artifact)
        data_validation_artifact = obj.main()
        print(data_validation_artifact.validation_status)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = 'Data Tranformation Stage'
try:
        if data_validation_artifact.validation_status:
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                obj = DataTransformationPipeline(data_validation_artifact)
                data_transformation_artifact = obj.main()
                print(data_transformation_artifact.best_features)
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
                print("The data has drift so please take a different data")
                logger.info("There is a drift in data")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = 'Model Training Stage'
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline(data_transformation_artifact)
        model_training_artifact = obj.main()
        print(model_training_artifact.metrics)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

