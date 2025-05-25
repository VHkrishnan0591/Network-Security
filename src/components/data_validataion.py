from src.utils.common import *
import pandas as pd
from scipy.stats import ks_2samp
from src.entity.config_entity import DataValidationArtifact
class DataValidation():
    def __init__(self,data_validation_config,data_ingestion_artifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
    
    def schema_validation(self,dataframe):
        schema = read_yaml(self.data_validation_config.schema_filepath)
        column_info = [{col:str(dtype)} for col, dtype in dataframe.dtypes.items()]
        if len(schema.columns) == len(column_info):
            for i in range(len(schema.columns)):
                key1, values1 = list(schema.columns[i].items())[0]
                key, value = list(column_info[i].items())[0]
                if (key == key1)  and (value ==values1):
                    validation_status = True
        return validation_status
    
    def check_for_data_drift(self, base_dataframe,dataframe):
        status=True
        report={}
        for column in base_dataframe.columns:
            d1=base_dataframe[column]
            d2=dataframe[column]
            is_same_dist=ks_2samp(d1,d2)
            if self.data_validation_config.data_validation_threshold<=is_same_dist.pvalue:
                is_found=False
            else:
                is_found=True
                status=False
            report.update({column:{
                "p_value":float(is_same_dist.pvalue),
                "drift_status":is_found
                }})
        write_yaml(report,self.data_validation_config.drift_report_filepath)
        return status


    
    def validating_data(self):

        # Creating the necessary directories
        logger.info("Creating the data validation directory")
        create_directories(self.data_validation_config.data_validation_directory)
        logger.info("Creating the valid data directory")
        create_directories(self.data_validation_config.valid_data_directory)
        logger.info("Creating the invalid data directory")
        create_directories(self.data_validation_config.invalid_data_directory)

        train_data = pd.read_csv(self.data_ingestion_artifact.train_filepath)
        test_data = pd.read_csv(self.data_ingestion_artifact.test_filepath)

        # Checking the schema for train data and test data
        logger.info("Checking the schema for train and test data")
        validation_status = self.schema_validation(train_data)
        validation_status = validation_status & self.schema_validation(test_data)
        logger.info("Checked the schema for train and test data")

        # Checking the datadrift for train data and test data
        logger.info("Checking the datadrift for train and test data")
        if validation_status:
            validation_status = self.check_for_data_drift(train_data,test_data)
        logger.info("Checked the datadrift for train and test data")

        if validation_status:
            train_data.to_csv(self.data_validation_config.valid_train_data_filepath, index=False)
            test_data.to_csv(self.data_validation_config.valid_test_data_filepath, index=False)
        else:
            train_data.to_csv(self.data_validation_config.invalid_train_data_filepath, index=False)
            test_data.to_csv(self.data_validation_config.invalid_test_data_filepath, index=False)
        return DataValidationArtifact(
            validation_status = validation_status,
            valid_train_data_filepath = self.data_validation_config.valid_train_data_filepath,
            valid_test_data_filepath = self.data_validation_config.valid_test_data_filepath,
            invalid_train_data_filepath = self.data_validation_config.invalid_train_data_filepath,
            invalid_test_data_filepath = self.data_validation_config.invalid_test_data_filepath,
            drift_report_filepath = self.data_validation_config.drift_report_filepath
        )
        
