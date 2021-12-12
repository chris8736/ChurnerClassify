from src.classifiers.xgboost import XGBoost
from src.preprocess import Preprocess
from src.data import Data

if __name__ == "__main__":
    data = Data()

    c = XGBoost()
    c.fit(data.training.X, y=data.training.y)
    y = c.predict(c.testing.X, y=c.testing.y)
    print(y)


    # p = Preprocess(filepath="data/training_stratified_80.csv").code_output()
    # p.combine_features(["Total_Relationship_Count","Credit_Limit","Customer_Age"], lambda x,y: x*y)
    # print(p.get_correlation("Total_Relationship_Count"))

    # c = XGBoost()
    # print(c.cross_validate("data/training_stratified_80.csv"))

    # c.train("data/training_stratified_80.csv")
    # c.predict("data/testing_stratified_20.csv")

    # e = Evaluate(c)
    # e.visualize()
