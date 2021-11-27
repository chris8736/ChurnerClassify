from .classifier import Classifier
from ..preprocess import Preprocess
from sklearn import svm
from sklearn.model_selection import cross_val_score


class SVM(Classifier):
    def __init__(self):
        self.c = svm.SVC()
        self.pp = None
        self.ready = False

    def preprocess(self, filepath):
        if self.pp is None:
            self.pp = Preprocess(filepath=filepath)
        else:
            self.pp.set_data(filepath=filepath)

        self.pp.code_output().onehot_encode()
        return self.pp.df, self.pp.y

    def cross_validate(self, filepath, cv):
        X, y = self.preprocess(filepath)
        return cross_val_score(
            self.c, X, y, cv=cv, scoring='roc_auc')

    def train(self, filepath):
        X, y = self.preprocess(filepath)
        self.ready = True
        self.c.fit(X, y)

    def predict(self, filepath):
        if not self.ready:
            raise Exception(
                "Please train the classifier before trying to predict")
        X, _ = self.preprocess(filepath)
        return self.c.predict(X)
