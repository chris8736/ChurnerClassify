from .classifier import Classifier
from xgboost import XGBClassifier

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
        
        # Initialize super class
        super().__init__()

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
    
    def preprocess(self, X=None, y=None, training=False):
        super().preprocess(X=X, y=y, training=training)

        self.pp.remove_columns().code_output().onehot_encode()
        return self.pp.df, self.pp.y
