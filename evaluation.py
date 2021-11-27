from src.evaluate import Evaluate
from src.classifiers.xgboost import XGBoost
from src.classifiers.svm import SVM
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class Evaluator():
    def __init__(self):
        return

    def cross_validate(self, clf, n_times=1):
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
        scores = []
        for clf in clfs:
            scores.append(self.cross_validate(clf, ntimes))
        return scores

    def n_train():
        return

    def n_test():
        return


if __name__ == "__main__":
    clfs = [XGBoost(n_estimators=100), SVM()]
    e = Evaluator()

    scores = e.n_cross_validate(clfs, 1)
    sns.barplot(data=scores)
    #plt.ylim(.99, 1)
    plt.show()

    c.train("data/training_stratified_80.csv")
    print("Finished training.\n")

    c.predict("data/testing_stratified_20.csv")
