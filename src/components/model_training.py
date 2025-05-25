from src.utils.common import *
import pandas as pd
from src.entity.config_entity import ModelTrainingArtifact
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class ModelTraining():
    def __init__(self,data_transformation_artifact,model_training_config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_config = model_training_config
        train_dataframe = pd.read_csv(self.data_transformation_artifact.transformed_test_data_filepath)
        test_dataframe = pd.read_csv(self.data_transformation_artifact.transformed_test_data_filepath)
        self.X_train = train_dataframe.drop(columns=self.model_training_config.target_column,axis=1)
        self.y_train = train_dataframe[self.model_training_config.target_column]
        self.X_test = test_dataframe.drop(columns=self.model_training_config.target_column,axis=1)
        self.y_test = test_dataframe[self.model_training_config.target_column]
        self.metrics= {}
    
    def evaluate_models(self, y_pred,model_name):
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        logger.info(f"\n===== {model_name} =====")
        logger.info("Accuracy: %s", (accuracy_score(self.y_test, y_pred)))
        logger.info("Confusion Matrix:\n %s", (confusion_matrix(self.y_test, y_pred)))
        logger.info("F1 Score: %s", f1)
        logger.info("Recall: %s", recall)
        logger.info("Classification Report:\n %s", (classification_report(self.y_test, y_pred)))
        self.metrics[model_name]= {'Accuracy':accuracy_score(self.y_test, y_pred),'Confusion Matrix': confusion_matrix(self.y_test, y_pred),
                        "F1 Score": f1, "Recall": recall, "Classification Report":classification_report(self.y_test, y_pred)
                        }
        # return self.metrics
        
    
    def hyper_parameter_tuning(self,model_name):
        best_models = {}
        if model_name == 'Logistic Regression':
            param_logreg = {
                'C': self.model_training_config.C,
                'penalty':self.model_training_config.penalty,  
                'solver': self.model_training_config.solver
            }
            grid_logreg = GridSearchCV(LogisticRegression(max_iter=self.model_training_config.max_iter), param_logreg, cv=self.model_training_config.cv, scoring=self.model_training_config.scoring)
            grid_logreg.fit(self.X_train, self.y_train)
            best_models['Logistic Regression'] = grid_logreg.best_estimator_

        elif model_name =='Decision Tree':
            param_tree = {
                'max_iter': self.model_training_config.DecisionTree_max_iter,
                'max_depth': self.model_training_config.DecisionTree_max_depth,
                'min_samples_split': self.model_training_config.DecisionTree_min_samples_split,
                'criterion': self.model_training_config.DecisionTree_criterion
            }
            grid_tree = GridSearchCV(DecisionTreeClassifier(), param_tree,cv=self.model_training_config.cv, scoring=self.model_training_config.scoring)
            grid_tree.fit(self.X_train, self.y_train)
            best_models['Decision Tree'] = grid_tree.best_estimator_

        elif model_name == 'Random Forest':
            param_rf = {
                'n_estimators': self.model_training_config.RandomForrest_n_estimators,
                'max_depth': self.model_training_config.RandomForrest_max_depth,
                'min_samples_split': self.model_training_config.RandomForrest_min_samples_split,
                'criterion': self.model_training_config.RandomForrest_criterion
            }
            grid_rf = GridSearchCV(RandomForestClassifier(), param_rf, cv=self.model_training_config.cv, scoring=self.model_training_config.scoring)
            grid_rf.fit(self.X_train, self.y_train)
            best_models['Random Forest'] = grid_rf.best_estimator_

        elif model_name =='Gradient Boosting':
            param_gb = {
                'n_estimators': self.model_training_config.GradientBoosting_n_estimators,
                'learning_rate': self.model_training_config.GradientBoosting_learning_rate,
                'max_depth': self.model_training_config.GradientBoosting_max_depth
            }
            grid_gb = GridSearchCV(GradientBoostingClassifier(), param_gb, cv=self.model_training_config.cv, scoring=self.model_training_config.scoring)
            grid_gb.fit(self.X_train, self.y_train)
            best_models['Gradient Boosting'] = grid_gb.best_estimator_

        elif model_name == 'AdaBoost':
            param_ada = {
                'n_estimators': self.model_training_config.AdaBoost_n_estimators,
                'learning_rate': self.model_training_config.AdaBoost_learning_rate
            }
            grid_ada = GridSearchCV(AdaBoostClassifier(), param_ada, cv=self.model_training_config.cv, scoring=self.model_training_config.scoring)
            grid_ada.fit(self.X_train, self.y_train)
            best_models['AdaBoost'] = grid_ada.best_estimator_
    
        for model_name, model in best_models.items():
            y_pred = model.predict(self.X_test)
            self.evaluate_models(y_pred,model_name+"HyperParameter_tuned")
            # metrics = self.evaluate_models(y_pred,model_name+"HyperParameter_tuned")
            logger.info("Saving the best Model")
            save_objects(model, self.model_training_config.best_model_filepath)

    def model_training(self):
        logger.info("Creating model training directory")
        create_directories(self.model_training_config.model_training_directory)
        logger.info("Reading the transformed test and train data")
        logger.info("Training the model")
        for model_name in self.model_training_config.list_of_models:
            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest": 
                model = RandomForestClassifier()
            elif model_name == "Gradient Boosting": 
                model = GradientBoostingClassifier()
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.evaluate_models(y_pred,model_name)
            # metrics = self.evaluate_models(y_pred,model_name)
        list_of_recall_score = {}
        max =0
        for model_name, metric_values in self.metrics.items():
            if metric_values['Recall'] > max:
                best_model_name = model_name
                max = metric_values['Recall']
        list_of_recall_score[best_model_name] = max
        logger.info("The best model to be tuned %s", best_model_name)
        self.hyper_parameter_tuning(best_model_name)
        logger.info("Logging the metrics to a metrics file")
        write_yaml({datetime.now():self.metrics},self.model_training_config.model_metrics_filepath)

        return ModelTrainingArtifact(
            best_model_filepath = self.model_training_config.best_model_filepath,
            model_metrics_filepath = self.model_training_config.model_metrics_filepath,
            metrics = self.metrics
        )

            