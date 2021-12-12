from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

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
        data = pd.read_csv(filepath)
        y = data.pop('Attrition_Flag') 
        X = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        self.training = DataSet(X_train, y_train,
            pd.concat([X_train, y_train], axis=1))
        self.testing = DataSet(X_test, y_test,
            pd.concat([X_test, y_test], axis=1))
        self.full = DataSet(
            pd.concat([self.training.X, self.testing.X]),
            pd.concat([self.training.y, self.testing.y]),
            pd.concat([self.training.merged, self.testing.merged]),
        )
