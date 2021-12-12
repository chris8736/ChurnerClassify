from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator
from ..preprocess import Preprocess

class Classifier (BaseEstimator):
    def __init__(self):
        self.c = None
        self.pp = None
        self.ready = False

    def preprocess(self, X=None,y=None, training=False):
        if self.pp is None or training:
            self.pp = Preprocess(X=X, y=y)
        else:
            self.pp.set_data(X=X, y=y)
        
        return self.pp.df, self.pp.y 

    def fit(self, X, y=None):
        X, y = self.preprocess(X=X, y=y, training=True)
        self.c.fit(X,y)
        self.ready = True

        return self

    def predict(self, X):
        if not self.ready:
            raise Exception(
                "Please train the classifier before trying to predict.")
        X = self.preprocess(X=X)

        return self.c.predict(X)

    def cross_validate(self, data, n=1):
        scores = []
        for i in range(n):
            s = cross_val_score(self, data.X, y=data.y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True),
                scoring=self.auc_precision_recall)
            scores.append(s)
        return scores

    @staticmethod
    def auc_precision_recall(estimator, X, y):
        return 1.0