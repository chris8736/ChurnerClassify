from functools import partial
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from scipy import stats

from re import findall

class Preprocess:
    """DataFrame wrapper class to preprocess data.
    All methods beginning with understand are require to take X and y as the last
    two positional arguments, even if one is not used. They must also return X, y

    All other methods return self, unless stated otherwise.
    """

    def __init__(self, use_default = True):
        """Instantiates a new Preprocess pipeline

        Parameters
        ----------
        use_default : boolean, default=True
            Use default pipeline.
        """
        self.pipeline = []
        if use_default:
            self.reindex().code_output().remove_columns().onehot_encode()
        
        self.oh_enc = None
        self.oh_cols = ['Gender', 'Education_Level', 'Marital_Status','Income_Category', 'Card_Category']
        self.correlated_features = []
        self.low_impact_features = []
        self.pca = None
        self.scaled = False
        self.scaler = None
    
    def execute(self, X, y, overwrite=False):
        """Runs the built pipeline against X,y and returns the preprocessed frames
        """
        if overwrite:
            self.oh_enc = None
            self.correlated_features=[]
            self.low_impact_features=[]
            self.pca = None
            self.scaler = None
        self.scaled = False
        for f in self.pipeline:
            X,y = f(X,y)
        
        return X,y

    def add(self, f):
        self.pipeline.append(f)
        return self
    
    def code_output(self):
        """Converts output vector into 0 or 1
        """
        return self.add(partial(self._code_output))
    def _code_output(self, X, y):
        if y is None:
            return x, None
        
        code = lambda x: 1 if x == "Attrited Customer" else 0
        return X, y.apply(code)

    def reindex(self):
        """Reindex the DataFrame
        """
        return self.add(partial(self._reindex))
    def _reindex(self, X, y):
        return X.reset_index(drop=True), y.reset_index(drop=True) if y is not None else None

    def remove_columns(self, columns = ["CLIENTNUM"]):
        """Removes columns from the DataFrame.

        Parameters
        ----------
        columns: List[str], default=["CLIENTNUM"]
           Columns to remove from the DataFrame
        """
        return self.add(partial(self._remove_columns, columns))
    def _remove_columns(self, columns, X, y):
        # Check that all columns exist in the DataFrame
        if not set(columns).issubset(X.columns):
            no_exist = [c for c in columns if c not in X.columns]
            raise LookupError(
                f"One or more columns were not found in the DataFrame: {no_exist}")

        return X.drop(columns, axis=1), y

    def onehot_encode(self, drop="first"):
        """Encodes categorical columns using one-hot encoding

        Parameters
        ----------
        drop : str, default="first"
            The drop policy for the OneHotEncoder.
            For more details, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        return self.add(partial(self._onehot_encode, drop))
    def _onehot_encode(self, drop, X, y):
        if self.oh_enc is None:  # First call
            self.oh_enc = OneHotEncoder(drop=drop)
            self.oh_enc.fit(X[self.oh_cols])

        onehot_df = pd.DataFrame(
            self.oh_enc.transform(X[self.oh_cols]).toarray(),
            columns=self.oh_enc.get_feature_names_out(self.oh_cols)
        ).astype(int)
        X,y = self._remove_columns(self.oh_cols,X,y)

        return pd.concat([X, onehot_df], axis=1), y

    def remove_correlated_features(self, threshold=0.9):
        """Removes features that have a correlation above the threshold
        Memorizes correlated features.

        Parameters
        ----------
        threshold: float, default=0.9
            The maximum value for correlation before removing the feature
        """
        return self.add(partial(self._remove_correlated_features, threshold))
    def _remove_correlated_features(self, threshold, X, y):
        if len(self.correlated_features) == 0:
            corr = X.corr().abs()  # Gets symmetrical square matrix
            # Select triangular matrix
            corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

            self.correlated_features.extend(
                [column for column in corr if any(corr[column] > threshold)])

        return self._remove_columns(self.correlated_features,X,y)

    def remove_low_impact(self, threshold=0.01):
        """Removes features with low correlation to the target
        Memorizes low impact features.

        Parameters
        ----------
        threshold: float, default=0.01
            The minimum value of correlation to the target before removing the feature.
        """
        return self.add(partial(self._remove_low_impact, threshold))
    def _remove_low_impact(self, threshold, X, y):
        output_column = "Attrition_Flag"
        if len(self.low_impact_features) == 0:  # First call
            df = pd.concat([X, y], axis=1)
            corr = df.corr().abs()
            corr_y = pd.DataFrame(corr[output_column]).sort_values(
                by=output_column)
            corr_y = corr_y[(corr_y[output_column]) <= threshold].drop([output_column], axis=1)
            self.low_impact_features.extend(corr_y.index.tolist())
        return self._remove_columns(self.low_impact_features, X, y)

    def drop_outliers(self, threshold=3):
        """Drops rows that are considered outliers

        Parameters
        ----------
        threshold: float, default=3
            The maximum magnitude of a zscore before removal
        """
        return self.add(partial(self._drop_outliers, threshold))
    def _drop_outliers(self, threshold, X, y):
        df, _ = self.extract_categorical(X)
        idx = (np.abs(stats.zscore(df)) < threshold).all(axis=1)

        return X[idx], y[idx]
        
    def convert_to_pcs(self, ignoreCategorical=True):
        """Convert df into its principal components.

        Parameters
        ----------
        ignoreCategorical: bool, default = True
            Whether to ignore categorical data or not.
        """
        return self.add(partial(self._convert_to_pcs, ignoreCategorical))
    def _convert_to_pcs(self, ignoreCategorical, X, y):
        if not self.scaled: # Data has not been scaled
            X,y = self._scale("minmax", X, y)

        # df, cat_df = self.extract_categorical(X)

        if self.pca is None:
            self.pca = PCA(n_components='mle')
            self.pca.fit(X)
        
        pc = self.pca.transform(X)
        pc_columns = ["PC" + str(i) for i in range(1, len(pc[0])+1)]
        pc_df = pd.DataFrame(data=pc, columns=pc_columns)

        # X = df if not ignoreCategorical else pd.concat([pc_df, cat_df], axis=1)
        return pc_df, y

    def scale(self, method="standard"):
        """Scale the data

        Parameters
        ----------
        method: {"standard", "minmax"}, default="standard"
            The scaler to use to scale the data
        """
        return self.add(partial(self._scale, method))
    def _scale(self, method, X, y):
        if self.scaled:
            return X, y

        if self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid method for scaling")
            self.scaler.fit(X)
        self.scaled = True
        scaled = self.scaler.transform(X)
        X = pd.DataFrame(scaled, index=X.index, columns=X.columns)
        return X, y

    def reduce_features(self, features, reducer, output="", sep="|"):
        """Reduces the given features into another.
        Does not remove the reduced features!

        Parameters
        ----------
        features: List[str]
            The features to be reduced.
        reducer: Callable(x,y) -> z
            A reducer lambda function that combines two values.
        output: str, default=""
            The name of the output feature. If left empty, it will simply
            concatenate the input features.
        sep: str, default="|"
            If using concatenation, this is the separator.
        """
        return self.add(partial(self._reduce_features, features, reducer, output, sep))
    def _reduce_features(self, features, reducer, output, sep, X, y): 
        name = output if len(output) > 0 else sep.join(features)
        input_df = X[features]
        output_df = reducer(input_df[features[0]], input_df[features[1]])
        for feature in features[2:]:
            output_df = reducer(output_df, input_df[feature])

        X[name] = output_df
        return X, y
    
    def map_features(self, features, mapper, output=""):
        """Maps the given features onto new features
        
        Parameters
        ----------
        features: List[str]
            The features to be mapped from
        mapper: Callable(x) -> y
            A mapper lambda function that maps from x to y
        output: str, default=""
            A str to append to the new feature name. If left empty, it will perform
            the mapping in-place
        """
        return self.add(partial(self._map_features, features, mapper, output))
    def _map_features(self, features, mapper, output, X, y):
        for feature in features:
            if len(output) == 0:
                X[feature] = X[feature].apply(mapper)
            else:
                X[feature+output] = X[feature].apply(mapper)
        return X,y
    
    ######################
    ### HELPER METHODS ###
    ######################
    def extract_categorical(self, df):
        """Extracts the non-categorical and the categorical data from self.df
        Note: this method does not return self!

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame with categorical data and non-categorical data.

        Returns
        -------
        df: pandas.DataFrame
            The non-categorical part of df
        cat_df: pandas.DataFrame
            The categorical part of df
        """
        if self.oh_enc is None:
            raise ReferenceError(
                "Please onehot encode the categorical data first!")

        features = self.oh_enc.get_feature_names_out(self.oh_cols)
        return df.drop(features, axis=1), df[features]

    def get_correlation(self, feature, X, y):
        """Get the correlation between the features and the output

        Parameters
        ----------
        feature: str
            The feature to retrieve correlation of
        X: the feature matrix
        y: the output vector
        """
        if feature not in X.columns:
            raise ReferenceError(f'Feature {f} is not a column of the data')
        
        return X[feature].corr(y)
   
    def __str__(self):
        buffer = []
        for f in self.pipeline:
            func = findall(r"Preprocess._.+?\b", str(f.func))
            args = str(f.args)
            buffer.append(f"{func}{args}")
        
        return "->".join(buffer)