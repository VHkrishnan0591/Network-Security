from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig():
    data_ingestion_directory: Path
    raw_data_folder: Path
    raw_data_file:Path
    split_data_folder: Path
    train_filepath: Path
    test_filepath: Path
    test_train_split:float
    database_name: str
    collection_name: str

@dataclass(frozen=True)
class DataIngestionArtifact():
    train_filepath: Path
    test_filepath: Path

@dataclass(frozen=True)
class DataValidationConfig():
    data_validation_directory: Path
    valid_data_directory: Path
    invalid_data_directory: Path
    valid_train_data_filepath: Path
    valid_test_data_filepath: Path
    invalid_train_data_filepath: Path
    invalid_test_data_filepath: Path
    drift_report_filepath: Path
    schema_filepath: Path
    data_validation_threshold:int

@dataclass(frozen=True)
class DataValidationArtifact():
    validation_status: str
    valid_train_data_filepath: Path
    valid_test_data_filepath: Path
    invalid_train_data_filepath: Path
    invalid_test_data_filepath: Path
    drift_report_filepath: Path

@dataclass(frozen=True)
class DataTransforamtionConfig():
    data_transforamtion_directory: Path
    model_directory:Path
    knn_imputer_model_filepath:Path
    feature_selection_model_filepath:Path
    transformed_train_data_filepath: Path
    transformed_test_data_filepath: Path
    n_neighbors:int
    missing_values:str
    target_column:str
    k_value:int

@dataclass(frozen=True)
class DataTransformationArtifact():
    best_features: list
    transformed_train_data_filepath: Path
    transformed_test_data_filepath: Path
    feature_selction_model:Path
    imputer_model:Path
@dataclass(frozen=True)
class ModelTrainerConfig():
    model_training_directory: Path
    best_model_filepath: Path
    target_column: str
    list_of_models:list
    max_iter:int
    C: list
    penalty: list
    solver: list
    DecisionTree_max_iter: int
    DecisionTree_max_depth: list
    DecisionTree_min_samples_split: list
    DecisionTree_criterion: list
    RandomForrest_n_estimators: list
    RandomForrest_max_depth: list
    RandomForrest_min_samples_split: list
    RandomForrest_criterion: list
    GradientBoosting_n_estimators: list
    GradientBoosting_learning_rate: list
    GradientBoosting_max_depth: list
    AdaBoost_n_estimators: list
    AdaBoost_learning_rate: list
    cv:int
    scoring:str
    model_metrics_filepath:Path
@dataclass(frozen=True)
class ModelTrainingArtifact:
    best_model_filepath: Path
    model_metrics_filepath: Path
    metrics: dict
