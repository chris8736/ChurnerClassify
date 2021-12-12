from .classifier import Classifier
from ..preprocess import Preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


class NeuralNet(Classifier):
    def __init__(self):
        self.c = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 2), random_state=1, max_iter=200)
        self.pp = None
        self.ready = False

    def preprocess(self, filepath):
        if self.pp is None:
            self.pp = Preprocess(filepath=filepath)
        else:
            self.pp.set_data(filepath=filepath)

        self.pp.code_output().onehot_encode()
        self.pp.scale()
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
