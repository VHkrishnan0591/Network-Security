
# importing module
import logging
from src.constants import LOG_DIRECTORY_PATH
import os

# Creating a folder
filepath = LOG_DIRECTORY_PATH
os.makedirs(filepath,exist_ok=True)
filename = os.path.join(filepath,"logs.log")

# Create and configure logger
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")

# Creating an object
logger = logging.getLogger(__name__)

# Adding the filehandler and console handler to the logger with a date time format
logger.addHandler(console_handler)
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt="%Y-%m-%d %H:%M",)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# Test messages
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")
# try:
#     x = 1 / 0
# except Exception as e:
#     logger.exception("Caught an exception: %s", e)
# logger.info("Finished")