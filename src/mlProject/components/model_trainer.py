import pandas as pd
import os
from mlProject import logger 
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Use ravel() to convert the column-vector y to a 1-dimensional array
        train_y = train_y.values.ravel()
        test_y = test_y.values.ravel()

        # Use check_array to ensure that y_train and y_test are 1-dimensional
        train_y = check_array(train_y, ensure_2d=False)
        test_y = check_array(test_y, ensure_2d=False)



        model = LogisticRegression(penalty=self.config.penalty, solver=self.config.solver, random_state=42)
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name)) 