from src.evaluate import Evaluate
from src.classifiers.xgboost import XGBoost

if __name__ == "__main__":
    c = XGBoost()
    c.train("data/training_stratified_80.csv")
    c.predict("data/testing_stratified_20.csv")