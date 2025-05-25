from src.entity.config_entity import DataIngestionConfig, DataTransforamtionConfig, DataValidationConfig,ModelTrainerConfig
from src.utils.common import create_directories, read_yaml
from src.constants import *

class ConfigurationManager():
    def __init__(self):
        self.config = read_yaml(CONFIG_FILEPATH)
        self.params = read_yaml(PARAMS_FILEPATH)
    
    def get_data_ingestion_config(self):
        return DataIngestionConfig(
            data_ingestion_directory = self.config.data_ingestion.data_ingestion_directory,
            raw_data_folder = self.config.data_ingestion.raw_data_folder,
            raw_data_file = self.config.data_ingestion.raw_data_file,
            split_data_folder = self.config.data_ingestion.split_data_folder,
            train_filepath = self.config.data_ingestion.train_filepath,
            test_filepath = self.config.data_ingestion.test_filepath,
            test_train_split = self.params.test_train_ratio,
            database_name = self.config.data_ingestion.database_name,
            collection_name = self.config.data_ingestion.collection_name,
        )
    
    def get_data_validation_config(self):
        return DataValidationConfig(
            data_validation_directory = self.config.data_validation.data_validation_directory,
            valid_data_directory = self.config.data_validation.valid_data_directory,
            invalid_data_directory = self.config.data_validation.invalid_data_directory, 
            valid_train_data_filepath = self.config.data_validation.valid_train_data_filepath,
            valid_test_data_filepath = self.config.data_validation.valid_test_data_filepath,
            invalid_train_data_filepath = self.config.data_validation.invalid_train_data_filepath,
            invalid_test_data_filepath = self.config.data_validation.invalid_test_data_filepath,
            drift_report_filepath = self.config.data_validation.drift_report_filepath,
            schema_filepath = self.config.data_validation.schema_filepath,
            data_validation_threshold = self.params.data_validation_threshold
        )
    
    def get_data_transformation_config(self):
        return DataTransforamtionConfig(
            data_transforamtion_directory = self.config.data_transformation.data_transforamtion_directory,
            model_directory = self.config.data_transformation.model_directory,
            knn_imputer_model_filepath = self.config.data_transformation.knn_imputer_model_filepath,
            feature_selection_model_filepath = self.config.data_transformation.feature_selection_model_filepath,
            transformed_train_data_filepath = self.config.data_transformation.transformed_train_data_filepath,
            transformed_test_data_filepath = self.config.data_transformation.transformed_test_data_filepath,
            n_neighbors = self.params.n_neighbors,
            missing_values = self.params.missing_values,
            target_column = self.params.target_column,
            k_value = self.params.k_value
        )
    
    def get_model_training_config(self):
        return ModelTrainerConfig(
            model_training_directory = self.config.model_training.model_training_directory ,
            target_column = self.params.target_column,
            best_model_filepath = self.config.model_training.best_model_filepath,
            list_of_models = self.params.list_of_models,
            max_iter = self.params.max_iter,
            C = self.params.C,
            penalty = self.params.penalty,
            solver = self.params.solver,
            DecisionTree_max_iter = self.params.DecisionTree_max_iter,
            DecisionTree_max_depth = self.params.DecisionTree_max_depth,
            DecisionTree_min_samples_split = self.params.DecisionTree_min_samples_split,
            DecisionTree_criterion = self.params.DecisionTree_criterion,
            RandomForrest_n_estimators = self.params.RandomForrest_n_estimators,
            RandomForrest_max_depth = self.params.RandomForrest_max_depth,
            RandomForrest_min_samples_split = self.params.RandomForrest_min_samples_split,
            RandomForrest_criterion = self.params.RandomForrest_criterion,
            GradientBoosting_n_estimators = self.params.GradientBoosting_n_estimators,
            GradientBoosting_learning_rate = self.params.GradientBoosting_learning_rate,
            GradientBoosting_max_depth = self.params.GradientBoosting_max_depth,
            AdaBoost_n_estimators = self.params.AdaBoost_n_estimators,
            AdaBoost_learning_rate = self.params.AdaBoost_learning_rate,
            cv = self.params.cv,
            scoring = self.params.scoring,
            model_metrics_filepath = self.config.model_training.model_metrics_filepath 
        )