from src.classifiers.xgboost import XGBoost
from src.preprocess import Preprocess

if __name__ == "__main__":
    p = Preprocess(filepath="data/training_stratified_80.csv").code_output()
    p.combine_features(["Total_Relationship_Count","Credit_Limit","Customer_Age"], lambda x,y: x*y)
    print(p.get_correlation("Total_Relationship_Count"))

    # c = XGBoost()
    # print(c.cross_validate("data/training_stratified_80.csv"))

    # c.train("data/training_stratified_80.csv")
    # c.predict("data/testing_stratified_20.csv")

    # e = Evaluate(c)
    # e.visualize()
