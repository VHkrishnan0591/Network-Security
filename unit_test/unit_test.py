import pickle
import pandas as pd
import sys
from src.logging.logger import logger
from sklearn.metrics import accuracy_score
test_data = pd.read_csv(r'unit_test\unit_test.csv')
X_test = test_data[:5].drop(columns='Result',axis=1)
y_test = test_data[:5]['Result']
with open(r'artifacts\model_directory\best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
y_pred = loaded_model.predict(X_test)
model_accuracy = (accuracy_score(y_test, y_pred))

if model_accuracy >0.9:
        pass
else:
        logger.error("The accuracy is not upto the mark")
        sys.exit(1)