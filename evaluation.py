from src.classifiers.xgboost import XGBoost
from src.classifiers.svm import SVM
from src.classifiers.randomforest import RandomForest
from src.classifiers.gradientboost import GradientBoost
from src.classifiers.bagging import Bagging
from src.classifiers.lightgbm import LightGBM
from src.classifiers.neuralnet import NeuralNet
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class Evaluator():
    def __init__(self):
        return

    def cross_validate(self, clf, n_times=1):
        """Returns score list for a 5-fold cross-validation.

        Parameters
        ----------
        clf : Classifier
            Classifier to test.
        n_times : int
            Number of times to shuffle and run cv.
        """
        score_list = []
        for i in range(n_times):
            kf = KFold(shuffle=True)
            scores = clf.cross_validate("data/training_stratified_80.csv", kf)
            score_list = score_list + list(scores)
        print("\nResults of cross-validation 5-fold: ")
        print("Average AUC: \t", sum(scores) / len(scores))
        print("Stdev: \t", statistics.stdev(scores), "\n")
        return scores

    def n_cross_validate(self, clfs, ntimes=1):
        """Returns score matrix for multiple classifiers' 5-fold cross-validation.

        Parameters
        ----------
        clfs : Classifier[]
            Classifiers to test.
        n_times : int
            Number of times to shuffle and run cv for each classifier.
        """
        scores = []
        for clf in clfs:
            scores.append(self.cross_validate(clf, ntimes))
        return scores

    def n_train():
        return

    def n_test():
        return


if __name__ == "__main__":
    # , Bagging(), SVM(), RandomForest(), GradientBoost(), NeuralNet()]
    clfs = [XGBoost(n_estimators=100), XGBoost(n_estimators=500), XGBoost(n_estimators=1000),
            LightGBM(n_estimators=100), LightGBM(n_estimators=500), LightGBM(n_estimators=1000), ]
    e = Evaluator()

    scores = e.n_cross_validate(clfs, 3)
    sns.barplot(data=scores)
    plt.ylim(.965, .996)
    plt.show()

    c.train("data/training_stratified_80.csv")
    print("Finished training.\n")

    c.predict("data/testing_stratified_20.csv")
