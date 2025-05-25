import yaml
from pathlib import Path
import os
from box import ConfigBox
import pickle
from src.logging.logger import logger

def read_yaml(path_to_yaml: str) -> ConfigBox:
    try:
        with open(path_to_yaml, 'r') as file:
            content = yaml.safe_load(file)
            logger.info("The yaml file is read successfully")
            return ConfigBox(content)
    except Exception as e:
        logger.exception("Caught an exception: %s", e)

def create_directories(directory_path:str):
    try:
        os.makedirs(directory_path,exist_ok=True)
        logger.info(f"The directory is created on the path {directory_path}")
    except Exception as e:
        logger.exception("Caught an exception: %s", e)

def write_yaml(data:dict,path_to_yaml: str):
    try:
        with open(path_to_yaml, 'w') as f:
                yaml.dump(data, f)
                logger.info("The yaml file is written successfully")
    except Exception as e:
        logger.exception("Caught an exception: %s", e)

def save_objects(model:object, path_of_model: str):
    try:
        with open(path_of_model, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logger.exception("Caught an exception. %s",e)

def load_objects(path_of_model:str):
    try:
        with open(path_of_model, 'rb') as file:
            loaded_objects = pickle.load(file)
        return loaded_objects
    except Exception as e:
        logger.exception("Caught an exception. %s",e)