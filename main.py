from src.classifiers import Classifier, XGBoost, LightGBM, Bagging, GradientBoost, NeuralNet, RandomForest, SVM
from src.preprocess import Preprocess
from src.data import Data

from statistics import mean, stdev
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from dataclasses import dataclass


@dataclass
class pp_score:
    pp: Preprocess
    mean: float
    std_dev: float

    def __repr__(self):
        return f"PP: {self.pp.__str__()}, mean: {self.mean}, stdev: {self.std_dev}"


def select_pp(cls, n=2, pps=[]):
    data = Data()
    results = []
    idx = 1
    total = n*len(pps)
    for pp in pps:
        scores = []
        for i in range(n):
            print(f"Iteration: {idx}/{total}")
            cls.pp = pp
            scores.extend(cls.cross_validate(data.training))
            idx += 1

        results.append(pp_score(pp, mean(scores), stdev(scores)))

    results.sort(key=lambda pp: pp.mean, reverse=True)
    return results

@dataclass
class hp_score:
    c: Classifier
    params: dict
    mean: float
    std_dev: float
    def __repr__(self):
        return f"C: {self.c.__str__()}, params: {self.params}, mean: {self.mean}, stdev: {self.std_dev}"

def select_hyperparameters(classifiers, n=2):
    data = Data()
    results = []
    
    for (c, params) in classifiers:
        clf = GridSearchCV(c, params, scoring=Classifier.auc_precision_recall)
        clf.fit(data.training.X, data.training.y)

        results.extend([
            hp_score(c, params, mean, std)
            for mean, std, params in zip(
                clf.cv_results_["mean_test_score"],
                clf.cv_results_["std_test_score"],
                clf.cv_results_["params"]
            )
        ])

    return results

if __name__ == "__main__":
    classifiers = [(XGBoost(), {
        'n_estimators': [60,80]
    })]
    hp_results = select_hyperparameters(classifiers)
    for r in hp_results:
        print(r)
    # results = select_pp(XGBoost(), n=10, pps=[
    #     Preprocess(),
    #     # PCA Experiment
    #     # Preprocess().convert_to_pcs(ignoreCategorical=False),
    #     # Preprocess().convert_to_pcs(),
    #     # Average_Open_To_Buy
    #     # Preprocess().remove_columns(columns=["Avg_Open_To_Buy"])
    #     # Scalers
    #     # Preprocess().scale(),
    #     # Preprocess().scale(method="minmax"),
    # ])

    # for r in results:
    #     print(r)
