from src.classifiers import Classifier, XGBoost, LightGBM, Bagging, GradientBoost, NeuralNet, RandomForest, SVM
from src.preprocess import Preprocess
from src.data import Data

from statistics import mean, stdev
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from dataclasses import dataclass

@dataclass
class score:
    c: Classifier
    params: dict
    mean: float
    std_dev: float
    def __repr__(self):
        return f"C: {self.c.__str__()}, mean: {self.mean}, stdev: {self.std_dev},\nPP: {self.params['pp'].__str__()},\nParams: { {k:v for k,v in self.params.items() if k != 'pp'} }"

def select(classifiers, n=2):
    data = Data()
    results = []
    
    for (c, params) in classifiers:
        clf = GridSearchCV(c, params, scoring=Classifier.auc_precision_recall, verbose=3)
        clf.fit(data.full.X, data.full.y)

        results.extend([
            score(c, params, mean, std)
            for mean, std, params in zip(
                clf.cv_results_["mean_test_score"],
                clf.cv_results_["std_test_score"],
                clf.cv_results_["params"]
            )
        ])

    results.sort(key=lambda r: r.mean, reverse=True)
    return results

if __name__ == "__main__":
    # Different Preprocess pipelines
    pp_base = Preprocess()
    pp_pcs_cat = Preprocess().convert_to_pcs(ignoreCategorical=False)
    pp_pcs_no_cat = Preprocess().convert_to_pcs(ignoreCategorical=True)
    pp_scale_std = Preprocess().scale()
    pp_scale_mm = Preprocess().scale(method="minmax")

    grid = []
    grid.append((XGBoost(), {
        "pp": [pp_base, pp_pcs_cat, pp_scale_std],
        "learning_rate": [0.3, 1, 2]
    }))

    results = select(grid)
    print("=====================")
    print("Results of GridSearch")
    print("=====================")
    for i,r in enumerate(results):
        print(f"--------------------------\n{i+1}. {r}")

    # Get final metrics for the assignment
    best = results[0]
    best.c.set_params(**best.params)
    scores = best.c.cross_validate(Data().full, n=20)
    print(f"Results of 20x5-fold Cross Validation:\n\tmean:{mean(scores)}\tstdev:{stdev(scores)}")