from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from sklearn.base import BaseEstimator
from .preprocess import Preprocess

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm


class Classifier (BaseEstimator):
    def __init__(self, pp=None):
        self.c = None
        self.pp = pp
        self.ready = False

    def preprocess(self, X, y, training=False):
        if self.pp is None:
            self.pp = Preprocess()
        return self.pp.execute(X, y, overwrite=training)

    def fit(self, X, y=None):
        X, y = self.preprocess(X, y, training=True)
        self.c.fit(X,y)
        self.ready = True

        return self

    def predict(self, X):
        if not self.ready:
            raise Exception(
                "Please train the classifier before trying to predict.")
        X,_ = self.preprocess(X, None)

        return self.c.predict(X)

    def cross_validate(self, data, n=1):
        scores = []
        for i in range(n):
            s = cross_val_score(self, data.X, y=data.y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True),
                scoring=self.auc_precision_recall, error_score='raise')
            scores.extend(s.tolist())

        return scores
    
    def predict_proba(self, X):
        return self.c.predict_proba(X)

    @staticmethod
    def auc_precision_recall(estimator, X, y):
        X,y = estimator.preprocess(X, y=y)
        y_score = estimator.predict_proba(X)[:,1]
        precision, recall, _ = precision_recall_curve(y, y_score)
        return auc(recall, precision)

class XGBoost(Classifier):
    def __init__(self,
                 pp=None,
                 n_estimators=60,
                 eval_metric="logloss",
                 scale_pos_weight=5.2,
                 learning_rate=0.3,
                 max_depth=6,
                 gamma=0,
                 subsample=1,
                 colsample_bytree=1
                 ):
        
        # Initialize super class
        super().__init__(pp=pp)
        self.n_estimators=n_estimators
        self.eval_metric=eval_metric
        self.scale_pos_weight=scale_pos_weight
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.gamma=gamma
        self.subsample=1
        self.colsample_bytree=colsample_bytree
        self.use_label_encoder=False

        # Create classifier
        self.c = XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            scale_pos_weight=scale_pos_weight,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            subsample=1,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False
        )
class LightGBM(Classifier):
    def __init__(self, pp=None, n_estimators=100):
        super().__init__(pp=pp)
        self.n_estimators=100

        self.c = LGBMClassifier(n_estimators=n_estimators)
class Bagging(Classifier):
    def __init__(self, pp=None):
        super().__init__(pp=pp)
        self.c = BaggingClassifier()
class GradientBoost(Classifier):
    def __init__(self, pp=None):
        super().__init__(pp=pp)
        self.c = GradientBoostingClassifier()
class NeuralNet(Classifier):
    def __init__(self, pp=None):
        super().__init__(pp=pp)
        self.c = MLPClassifier(solver='lbfgs', alpha=1e-5,\
            hidden_layer_sizes=(5,2), max_iter=200)
class RandomForest(Classifier):
    def __init__(self, pp=None):
        super().__init__(pp=pp)
        self.c = RandomForestClassifier()
class SVM(Classifier):
    def __init__(self,pp=None):
        super().__init__(pp=pp)
        self.c = svm.SVC(probability=True)