import pickle
import pandas as pd
import sys
import os
from src.logging.logger import logger
from sklearn.metrics import accuracy_score
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'unit_test.csv')
model_path = os.path.join(base_dir, '..', 'artifacts', 'model_directory', 'best_model.pkl') 
test_data = pd.read_csv(csv_path) # r'unit_test\unit_test.csv'
X_test = test_data[:5].drop(columns='Result',axis=1)
y_test = test_data[:5]['Result']
with open(model_path, 'rb') as file: # r'artifacts\model_directory\best_model.pkl'
            loaded_model = pickle.load(file)
y_pred = loaded_model.predict(X_test)
model_accuracy = (accuracy_score(y_test, y_pred))

if model_accuracy >0.9:
        pass
else:
        logger.error("The accuracy is not upto the mark")
        sys.exit(1)