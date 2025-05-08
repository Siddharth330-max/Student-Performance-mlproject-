import os 
import sys 
from dataclasses import dataclass  

from sklearn.ensemble import(
    AdaBoostRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")   ##join to artifacts file and output willl be model.pkl

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()     ##got initialized to ModelTrainerConfig  
                                                             ##inside the variable we can get the file path name
        
    def initiate_model_trainer(self ,train_array,test_array):      ##Thses are outputs of data transformation
        try:
            logging.info("Splitting data into training and test input data")
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Nearest Neighbour Regressor" : KNeighborsRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor(),
                "XGBRegressor" : XGBRegressor(),
            }

            model_report:dict= evaluate_models(X_train = X_train , y_train = y_train,X_test = X_test ,y_test =  y_test,
                                               models = models)
            
            ##To get best model score from dict
            best_model_score = max(sorted(model_report.values()))  

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found")

            ##save_object to save the model path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model                       ##It will be created as model.pkl file
            )
            
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test , predicted)
            return r2_square
        except Exception as e :
            raise CustomException(e , sys)
