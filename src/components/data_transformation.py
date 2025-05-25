from src.utils.common import *
import pandas as pd
from src.entity.config_entity import DataTransformationArtifact
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer

class DataTransformation():
    def __init__(self, data_validation_artifact,data_transforamtion_config):
        self.data_validation_artifact = data_validation_artifact
        self.data_transforamtion_config = data_transforamtion_config
    
    def feature_selection(self, dataframe):
        dataframe[self.data_transforamtion_config.target_column] = dataframe[self.data_transforamtion_config.target_column].replace(-1,0)
        X = dataframe.drop(self.data_transforamtion_config.target_column,axis =1)
        Y = dataframe[self.data_transforamtion_config.target_column]
        # X: features, y: target
        selector = SelectKBest(score_func=f_classif, k=self.data_transforamtion_config.k_value)  # Select top 10 features
        X_new = selector.fit_transform(X, Y)

        logger.info("Saving the feature selection model")
        save_objects(selector,self.data_transforamtion_config.feature_selection_model_filepath)

        # Get selected feature indices or names
        selected_features = selector.get_support(indices=True)
        return [X.columns[i] for i in selected_features]
    
    def data_transformation(self):
        logger.info("Creating Data Transforamtion and model directory")
        create_directories(self.data_transforamtion_config.data_transforamtion_directory)
        create_directories(self.data_transforamtion_config.model_directory)
        logger.info("Reading the validated data")
        train_dataframe = pd.read_csv(self.data_validation_artifact.valid_train_data_filepath)
        test_dataframe = pd.read_csv(self.data_validation_artifact.valid_test_data_filepath)
        logger.info("Performing KNN Imputation")
        imputer = KNNImputer(n_neighbors=self.data_transforamtion_config.n_neighbors)
        imputed_train_dataframe = imputer.fit_transform(train_dataframe)
        imputed_test_dataframe = imputer.transform(test_dataframe)
        imputed_train_dataframe = pd.DataFrame(imputed_train_dataframe,columns=train_dataframe.columns)
        imputed_test_dataframe = pd.DataFrame(imputed_test_dataframe,columns=train_dataframe.columns)
        is_null = (imputed_train_dataframe.isnull().values.any()) & (imputed_train_dataframe.isnull().values.any())
        if is_null == False:
            logger.info("Saving the Imputer Mdoel")
            save_objects(imputer,self.data_transforamtion_config.knn_imputer_model_filepath)
            logger.info("Saving the imputed train and test dataframe")
            imputed_train_dataframe.to_csv(self.data_transforamtion_config.transformed_train_data_filepath, index=False)
            imputed_test_dataframe.to_csv(self.data_transforamtion_config.transformed_test_data_filepath, index=False)
            logger.info("Performing feature selction")
            best_features = self.feature_selection(imputed_train_dataframe)
            return DataTransformationArtifact(
                best_features = best_features,
                transformed_train_data_filepath = self.data_transforamtion_config.transformed_train_data_filepath,
                transformed_test_data_filepath = self.data_transforamtion_config.transformed_test_data_filepath,
                feature_selction_model = self.data_transforamtion_config.feature_selection_model_filepath,
                imputer_model = self.data_transforamtion_config.knn_imputer_model_filepath
            )
        else:
            logger.error("Imputer Failed")

