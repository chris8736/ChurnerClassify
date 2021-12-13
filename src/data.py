from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

@dataclass
class DataSet:
    X: pd.DataFrame
    y: pd.DataFrame
    merged: pd.DataFrame

class Data:
    """DataFrame wrapper class to split the data.
    This class will split the data into 80% training data
    and 20% testing data.
    """

    def __init__(self, filepath="data/full.csv"):
        """ Instantiates a new Data object

        Parameters
        ----------
        filepath : string, default="data/full.csv"
            File path of the file containing the data set
        """
        self.df = pd.read_csv(filepath)
        self.X = self.df.drop('Attrition_Flag', axis=1)
        self.y = self.df['Attrition_Flag']
    
    def train_test_split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)
        training = DataSet(X_train, y_train,
            pd.concat([X_train, y_train], axis=1))
        testing = DataSet(X_test, y_test,
            pd.concat([X_test, y_test], axis=1))
        full = DataSet(
            pd.concat([training.X, testing.X]),
            pd.concat([training.y, testing.y]),
            pd.concat([training.merged, testing.merged]),
        )

        return (training, testing, full)
       
    def kfolds(self, n_splits=5, df=None, y=None):
        df = self.df if df is None else df
        y = self.y if y is None else y
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True)
        return folds.split(df, y)

