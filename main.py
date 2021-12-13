from src.classifiers import Classifier, XGBoost, LightGBM, Bagging, GradientBoost, NeuralNet, RandomForest, SVM
from src.preprocess import Preprocess
from src.data import Data

from joblib import Parallel, delayed, parallel_backend

from statistics import mean, stdev
from sklearn.model_selection import ParameterGrid, GridSearchCV, StratifiedKFold
from dataclasses import dataclass

@dataclass
class score:
    c: Classifier
    params: dict
    mean: float
    std_dev: float
    def __repr__(self):
        return f"C: {self.c.__str__()}, mean: {self.mean}, stdev: {self.std_dev},\nPP: {self.params['pp'].__str__()},\nParams: { {k:v for k,v in self.params.items() if k != 'pp'} }"

@dataclass
class raw_score:
    c: Classifier
    p: dict
    score: float
    def __repr__(self):
        return f"C: {self.c.__str__()}, Score: {self.score}\nPP: {self.params['pp'].__str__()},\nParams: { {k:v for k,v in self.params.items() if k != 'pp'} }"

def select(grid, X, y):
    with parallel_backend('threading', n_jobs=16):
        scores = []
        for (c, params) in grid:
            clf = GridSearchCV(c, params, scoring=Classifier.auc_precision_recall, verbose=3, n_jobs=16)
            clf.fit(X, y)

            scores.extend([
                score(c, params, mean, std)
                for mean, std, params in zip(
                    clf.cv_results_["mean_test_score"],
                    clf.cv_results_["std_test_score"],
                    clf.cv_results_["params"]
                )
            ])
        scores.sort(key=lambda r: r.mean, reverse=True)
        return scores

def cross_validate_with_tuning(grid, n=20):
    # with Parallel(n_jobs=16, prefer="threads", verbose=100) as parallel:
    final_scores = []
    idx = 0
    for i in range(n):
        data = Data()
        for train_idx, test_idx in data.kfolds():
            X_train, X_test = data.X.iloc[train_idx], data.X.iloc[test_idx]
            y_train, y_test = data.y.iloc[train_idx], data.y.iloc[test_idx]
            df_train, df_test = data.df.iloc[train_idx], data.df.iloc[test_idx]

            # raw_scores = parallel(delayed(cvt_)(c, params, X_train, y_train, train_idx, test_idx) for c, grid_ in grid for params in ParameterGrid(grid_) for (train_idx, test_idx) in data.kfolds(df=df_train, y=y_train))
            # raw_scores = [s for sl in scores for s in sl]

            scores = select(grid, X_train, y_train)
            
            best = scores[0]
            final_scores.append(raw_score(best.c, best.params, Classifier.auc_precision_recall(best.c, X_test, y_test)))

    return final_scores
    
if __name__ == "__main__":
    # Different Preprocess pipelines
    pp_base = Preprocess()
    pp_pcs_cat = Preprocess().convert_to_pcs(ignoreCategorical=False)
    pp_pcs_no_cat = Preprocess().convert_to_pcs(ignoreCategorical=True)
    pp_scale_std = Preprocess().scale()
    pp_scale_mm = Preprocess().scale(method="minmax")

    xgboost_params = (XGBoost(), {
        "pp": [pp_base, pp_pcs_cat, pp_scale_std],
        "learning_rate": [0.1, 0.3, 0.5,],
        "max_depth": [5,6,7],
        "scale_pos_weight": [5,6,7],
        "min_child_weight": [1,3,5],
    })

    lightgbm_params = (LightGBM(), {
        "pp": [pp_scale_std],
        "learning_rate": [0.05, 0.03, 0.01],
        "n_estimators": [300, 400, 500],
        "num_leaves": [31, 63]
    })

    rs = cross_validate_with_tuning([lightgbm_params], n=20)
    print("====================================================================")
    print("Results of Nested 5-Fold Cross Validation with Hyperparameter Tuning")
    print("====================================================================")
    print("Models/parameters used: ")
    for i,r in enumerate(results):
        print(f"-------------------------\n{i+1}: {r}")
    
    print(f"Final Metrics: mean: {mean([s.score for s in rs])}, stdev: {stdev([s.score for s in rs])}")