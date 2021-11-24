from classifier import Classifier
from preprocess import Preprocess

class XGBoost(Classifier):
    def __init__(self, 
        n_estimators=60, 
        eval_metric="logloss", 
        scale_pos_weight=5.2,
        learning_rate=0.3
    ):
        self.c = XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            scale_pos_weight=scale_pos_weight,
            learning_rate=learning_rate
        )
    
    # Loads the data from given filepath
    # Preprocesses it and returns the feature matrix and the label
    def preprocess(self, filepath, training=False):
        ppc = Preprocess(filepath).code_output().drop_output().onehot_categorical()
        
        return (ppc.df, ppd.output)
    
    def train(self, filepath):
        (X, y) = self.preprocess(filepath, training=True)
        self.c.fit(X, y)

    def predict(self, filepath):
        (X, _y) = self.preprocess(filepath, training=False)
        return