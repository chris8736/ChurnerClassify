from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy import stats

# SKLearn Imports
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import math


class Preprocess:
    """DataFrame wrapper class to preprocess data.
    Unless stated otherwise, all non-static methods will return the instance they were called from.
    This is to facilitate method chaining.
    """

    def __init__(self, X=pd.DataFrame(), y=pd.DataFrame(), filepath: str = "", scaled=False):
        """Instantiates a new Preprocess object

        Parameters
        ----------
        X : pandas.DataFrame, default=[]
            Feature vector to preprocess
        y : pandas.DataFrama, default=[]
            Output vector associated with X
        filepath : str, default=""
            If not empty, will overwrite X and y with data read from the given path.
            Note: shorthand for loading data and then passing it through X and y.
        scaled : bool, default=False
            Whether the data is scaled.
        """
        self.set_data(X=X, y=y, filepath=filepath)

        self.output_column = "Attrition_Flag"

        # Memorization fields
        self.oh_enc = None                  # One Hot Encoder
        self.oh_cols = None                 # One Hot Columns
        self.correlated_features = []       # Correlated features to drop
        self.low_impact_features = None     # Low impact features to drop
        self.pca = None                     # PCA model
        self.scaler = None                  # Scaler

    def set_data(self, X=pd.DataFrame(), y=pd.DataFrame(), filepath: str = "", scaled=False):
        """ Set the data to preprocessed.

        Parameters
        ----------
        See the constructor's docstring for information.
        """
        self.df = X
        self.y = y

        if len(filepath) > 0:
            self.df, self.y = Preprocess.split_data(filepath)

        self.df = self.df.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        self.scaled = scaled

        return self

    @staticmethod
    def split_data(
        filepath: str,
        output_column: str = "Attrition_Flag",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads a CSV file into a pandas.DataFrame and then splits it into 
        feature matrix X and output vector y.

        Parameters
        ----------
        filepath: str
            Path of the file to load the data from.

        output_column : str, default="Attrition_Flag"
            The column from the pandas.DataFrame to extract as an output vector

        Returns
        -------
        X : pandas.DataFrame
            Feature matrix X
        y : pandas.DataFrame
            Output vector y
        """
        df = pd.read_csv(filepath)
        return (df.drop([output_column], axis=1), df[output_column])

    # ---------------- #
    # Data Structuring #
    # ---------------- #

    def remove_columns(self, columns: List[str] = ["CLIENTNUM"]) -> Preprocess:
        """Removes columns from the DataFrame.

        Parameters
        ----------
        columns: List[str], default=["CLIENTNUM"]
            Columns to remove from the DataFrame
        """
        # Check that all columns exist in the DataFrame
        if not set(columns).issubset(self.df.columns):
            no_exist = [c for c in columns if c not in self.df.columns]
            raise LookupError(
                f"One or more columns were not found in the DataFrame: {no_exist}")

        self.df.drop(columns, axis=1, inplace=True)
        return self

    def code_output(self, use_y=True) -> Preprocess:
        """Converts output vector into 0 or 1.

        Parameters
        ----------
        use_y : bool, default=True
            If True, processes self.y. Otherwise, it will process
            self.df[self.output_column] instead.
        """
        def code(x): return 1 if x == "Attrited Customer" else 0
        if use_y:
            if len(self.y) == 0:
                raise ValueError("Output vector y is empty.")

            self.y = self.y.apply(code)
        else:
            self.df[self.output_column] = self.df[self.output_column].apply(
                code)

        return self

    def onehot_encode(self,
                      columns=['Gender', 'Education_Level', 'Marital_Status',
                               'Income_Category', 'Card_Category'],
                      drop="first",
                      ) -> Preprocess:
        """Encodes categorical columns using one-hot encoding

        Parameters
        ----------
        columns : List[str], default=['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
            Columns to one-hot encode. This is memorized; subsequent calls to
            onehot_encode will use the same value.
        drop : str, default="first"
            The drop policy for the OneHotEncoder.
            For more details, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        if self.oh_enc is None:  # First call
            self.oh_enc = OneHotEncoder(drop=drop)
            self.oh_cols = columns
            self.oh_enc.fit(self.df[self.oh_cols])

        onehot_df = pd.DataFrame(
            self.oh_enc.transform(self.df[self.oh_cols]).toarray(),
            columns=self.oh_enc.get_feature_names_out(self.oh_cols)
        ).astype(int)
        self.remove_columns(columns=self.oh_cols)
        self.df = pd.concat(
            [self.df, onehot_df],
            axis=1
        )

        return self

    def shuffle(self):
        """Shuffles the DataFrame
        """
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self

    def extract_categorical(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts the non-categorical and the categorical data from self.df
        Note: this method does not return self!

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame with categorical data and continuous data.

        Returns
        -------
        df: pandas.DataFrame
            The continuous part of df
        cat_df: pandas.DataFrame
            The categorical part of df
        """
        if self.oh_enc is None:
            raise ReferenceError(
                "Please onehot encode the categorical data first!")

        features = self.oh_enc.get_feature_names_out(self.oh_cols)
        return self.df.drop(features, axis=1), self.df[features]

    # ------------------- #
    # Feature Engineering #
    # ------------------- #

    def remove_correlated_features(self, threshold=0.9, append=False):
        """Removes features that have a correlation above the threshold
        Memorizes correlated features.

        Parameters
        ----------
        threshold: float, default=0.95
            The maximum value for correlation before removing the feature
        """
        if len(self.correlated_features) == 0 or append:
            corr = self.df.corr().abs()  # Gets symmetrical square matrix
            # Select triangular matrix
            corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

            self.correlated_features.extend(
                [column for column in corr if any(corr[column] > threshold)])

        self.remove_columns(columns=self.correlated_features)

        return self

    def remove_low_impact(self, threshold=0.001):
        """Removes features with low correlation to the target
        Memorizes low impact features.

        Parameters
        ----------
        threshold: float, default=0.01
            The minimum value of correlation to the target before removing the feature.
        """
        if self.low_impact_features is None:  # First call
            df = pd.concat([self.df, self.y], axis=1)
            corr = df.corr().abs*()
            corr_y = pd.DataFrame(corr[self.output_column]).sort_values(
                by=self.output_column)
            self.low_impact_features = corr_y[(corr_y[self.output_column]) <= threshold]\
                .drop([self.output_column], axis=1)

        self.remove_columns(columns=self.low_impact_features)

        return self

    def drop_outliers(self, threshold=3):
        """Drops rows that are considered outliers

        Parameters
        ----------
        threshold: float, default=3
            The maximum magnitude of a zscore before removal
        """
        df, _ = self.extract_categorical()

        idx = (np.abs(stats.zscore(df)) < threshold).all(axis=1)
        self.df = self.df[idx]
        self.y = self.y[idx]

        return self

    def convert_to_pcs(self, ignoreCategorical=True, variance=0.95):
        """Convert df into its principal components.

        Parameters
        ----------
        ignoreCategorical: bool, default = True
            Whether to ignore categorical data or not.
        variance: [0,1], default=0.95
            The explained variance for PC selection
        """

        if not self.scaled:  # Data has not been scaled
            self.scale(method="minmax")

        df, cat_df = self.extract_categorical()

        if self.pca is None:
            self.pca = PCA(n_components='mle')
            self.pca.fit(df)

        pc = self.pca.transform(df)
        pc_columns = ["PC" + str(i) for i in range(1, len(pc[0])+1)]
        pc_df = pd.DataFrame(data=pc, columns=pc_columns)

        self.df = pd.concat([pc_df, cat_df], axis=1)
        return self

    def scale(self, method="standard"):
        """Scale the data

        Parameters
        ----------
        method: {"standard", "minmax"}, default="standard"
            The scaler to use to scale the data
        """
        if self.scaled:  # Data is already scaled
            return self

        if self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid method for scaling.")
            self.scaler.fit(self.df)

        self.scaled = True
        scaled_features = self.scaler.transform(self.df.values)
        self.df = pd.DataFrame(scaled_features,
                               index=self.df.index, columns=self.df.columns)
        return self
    
    def combine_features(self, features: List[str], reducer, output="", sep="|"):
        """Combine the given features into another.
        Does not remove the combined features!

        Parameters
        ----------
        features: List[str]
            The features to be combined.
        reducer: Callable(x,y) -> z
            A reducer lambda function that combines two values.
        output: str, default=""
            The name of the output feature. If left empty, it will simply
            concatenate the input features.
        sep: str, default="|"
            If using concatenation, this is the separator.
        """
        if len(features) < 2:
            return self

        name = output if len(output) > 0 else sep.join(features)
        input_df = self.df[features]
        output_df = reducer(input_df[features[0]], input_df[features[1]])
        for feature in features[2:]:
            output_df = reducer(output_df, input_df[feature])

        self.df[name] = output_df
        return self

    def get_correlation(self, feature):
        """Get the correlation between the feature and the output (y)
        DOES NOT RETURN SELF
        Parameters
        ----------
        feature: str
            The feature to retrieve correlation of.
        """
        if feature not in self.df.columns:
            raise ReferenceError(
                f'Feature {f} is not a column in the data'
            )
        
        return self.df[feature].corr(self.y)

    def graph_2d_pca(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        ax.set_facecolor('gray')
        ax.scatter(self.df['PC1'],
                   self.df['PC2'], c=self.output, cmap='gray')

        # for i in range(len(X)):
        #    ax.annotate(i, (principalDf['principal component 1'][i],
        #                    principalDf['principal component 2'][i]))

        ax.grid()
        plt.show()

    def graph_3d_pca(self):
        # plot pc1, pc2, and pc3 axes
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # labels
        ax.set_xlabel('Principal Component 1', fontsize=8)
        ax.set_ylabel('Principal Component 2', fontsize=8)
        ax.set_zlabel('Principal Component 3', fontsize=8)
        ax.set_title('3 component PCA', fontsize=20)
        ax.set_facecolor('gray')
        ax.scatter3D(self.df['PC1'],
                     self.df['PC2'],
                     self.df['PC3'], c=self.output, cmap='gray')

        ax.grid()
        plt.show()

    def add_product_features(self):
        features = self.df.columns
        for feature1 in features:
            for feature2 in features:
                self.df[feature1 + "_*_" + feature2] = self.df[feature1] * \
                    self.df[feature2]
                #print(feature1 + "_*_" + feature2)
        return self

    def add_quotient_features(self):
        features = self.df.columns
        for feature1 in features:
            for feature2 in features:
                self.df[feature1 + "_/_" + feature2] = self.df[feature1] / \
                    (self.df[feature2] + 1)
                #print(feature1 + "_*_" + feature2)
        return self

    def add_log_features(self):
        features = self.df.columns
        for feature in features:
            self.df["log_" + feature] = self.df.apply(math.log(feature))
