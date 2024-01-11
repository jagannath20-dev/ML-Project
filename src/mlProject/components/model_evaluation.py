import os
import pandas as pd 
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import joblib
import numpy as np
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        acc_score = accuracy_score(predicted_qualities,test_y)
        # Saving metrics as local
        scores = {"accuracry_score" : acc_score}
        save_json(path=Path(self.config.metric_file_name), data=scores)