from src.classifiers import XGBoost, LightGBM, Bagging, GradientBoost, NeuralNet, RandomForest, SVM
from src.preprocess import Preprocess
from src.data import Data

if __name__ == "__main__":
    data = Data()

    # pp = Preprocess().reduce_features(["Dependent_count", "Total_Relationship_Count"], lambda x,y: x*y)\
    #     .map_features(["Customer_Age"], lambda x: x*12, output="_in_months")
    # X,y=pp.execute(data.training.X, data.training.y)
    # Xt, yt = pp.execute(data.testing.X, data.testing.y)

    # print(X,y)

    pp = [
        Preprocess().remove_low_impact().remove_correlated_features(),
        Preprocess().scale(),
        Preprocess().convert_to_pcs()
    ]

    c = [
        XGBoost(), LightGBM(), Bagging(), GradientBoost(), RandomForest(), SVM()
    ]

    for i in range(len(pp)):
        for j in range(len(c)):
            c[j].pp = pp[i]
            scores = c[j].cross_validate(data.training)
            print(f"Classifier {j} with PP{i}: {scores}")