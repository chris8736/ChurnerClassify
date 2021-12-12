from .classifier import Classifier
from ..preprocess import Preprocess
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


class XGBoost(Classifier):
    def __init__(self,
                 n_estimators=60,
                 eval_metric="logloss",
                 scale_pos_weight=5.2,
                 learning_rate=0.3,
                 max_depth=6,
                 gamma=0,
                 subsample=1,
                 colsample_bytree=1
                 ):
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
