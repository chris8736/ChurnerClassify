from src.classifiers import XGBoost
from src.preprocess import Preprocess
from src.data import Data

if __name__ == "__main__":
    data = Data()

    # pp = Preprocess().reduce_features(["Dependent_count", "Total_Relationship_Count"], lambda x,y: x*y)\
    #     .map_features(["Customer_Age"], lambda x: x*12, output="_in_months")
    # X,y=pp.execute(data.training.X, data.training.y)
    # Xt, yt = pp.execute(data.testing.X, data.testing.y)

    # print(X,y)

    c1 = XGBoost()
    scores = c1.cross_validate(data.training)
    print(scores)