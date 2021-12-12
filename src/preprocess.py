from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from typing import List
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# This is a wrapper class to allow preprocessing data in different ways.
# To add a method to this class, just make sure it returns itself at the end.
# See remove_columns for an example


class Preprocess:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.df: pd.DataFrame
        self.output = None

        self.load_file()

    def load_file(self) -> None:
        self.df = pd.read_csv(self.filepath)

    # Remove columns from the dataset
    # By default this just removes the CLIENTNUM column.
    def remove_columns(self, columns: List[str] = ["CLIENTNUM"]) -> Preprocess:
        # Validate columns type
        if type(columns) != list:
            raise TypeError(
                f"Columns must be a list, but it is of type {type(columns)} instead.")

        # Check that all columns exist in the DataFrame
        if not set(columns).issubset(self.df.columns):
            no_exist = [c for c in columns if c not in self.df.columns]
            raise LookupError(
                f"One or more columns were not found in the DataFrame: {no_exist}")

        self.df.drop(columns, axis=1, inplace=True)
        return self

    # Data Structuring

    # make attrition flag 0s and 1s
    def code_output(self):
        self.df["Attrition_Flag"] = self.df["Attrition_Flag"].apply(
            lambda x: 1 if x == "Attrited Customer" else 0)
        return self

    def drop_output(self):
        self.output = self.df.pop("Attrition_Flag")
        return self

    # code categorical vaiables
    def onehot_categorical(self):
        self.df = pd.get_dummies(
            self.df, columns=['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
        return self

    def shuffle(self):
        self.df = self.df.sample(frac=1, random_state=0)
        return self

    # Feature Selection + Visualization

    # Remove highly correlated features from the dataset
    def remove_high_correlation(self, threshold):
        # get sorted correlation pairs
        c = self.df.corr().abs()
        s = c.unstack()
        so = s.sort_values(kind="quicksort")

        # remove from dataset if above threshold
        last_label1 = ""
        last_label2 = ""
        cols_to_remove = []
        for label, corr_value in zip(so.axes[0], so):
            if (label[0] == last_label2 and label[1] == last_label1):  # if pair is equal
                continue
            last_label1 = label[0]
            last_label2 = label[1]
            if (corr_value != 1 and corr_value > threshold):
                if (not label[0] in cols_to_remove):
                    cols_to_remove.insert(len(cols_to_remove), label[0])
        self.remove_columns(cols_to_remove)
        print("Removed the following features as correlation with another feature was >" +
              str(threshold) + ": " + str(cols_to_remove))
        return self

    def correlation_pairs(self):
        c = self.df.corr().abs()
        s = c.unstack()
        so = s.sort_values(kind="quicksort")
        return so

    def print_correlation_pairs(self):
        c = self.df.corr().abs()
        s = c.unstack()
        so = s.sort_values(kind="quicksort")
        for label, corr_value in zip(so.axes[0], so):
            if (corr_value != 1):
                row = (label[0], label[1], str(corr_value))
                print("{: >30} {: >30} {: >30}".format(*row))

    # Remove features with low correlation towards the output
    def remove_low_impact(self, threshold):
        # Correlation with output variable
        self.df["Output"] = self.output
        cor = self.df.corr()
        cor_target = abs(cor["Output"])
        so = cor_target.sort_values(kind="quicksort")

        cols_to_remove = cor_target[cor_target < threshold].index.tolist()
        self.remove_columns(cols_to_remove)
        print("Removed the following features as correlation with output was <" +
              str(threshold) + ": " + str(cols_to_remove))
        self.remove_columns(["Output"])
        return

    def output_correlation(self):
        self.df["Output"] = self.output
        cor = self.df.corr()
        cor_target = cor["Output"]
        so = cor_target.sort_values(kind="quicksort")
        self.remove_columns(["Output"])
        return so[:-1]

    def print_output_correlation(self):
        self.df["Output"] = self.output
        cor = self.df.corr()
        cor_target = cor["Output"]
        so = cor_target.sort_values(kind="quicksort")
        self.remove_columns(["Output"])
        print(so[:-1])

    def drop_outliers(self, zthreshold):
        # need this so output gets changed accordingly
        self.df["Output"] = self.output
        self.df = self.df[(np.abs(stats.zscore(self.df))
                           < zthreshold).all(axis=1)]
        self.output = self.df.pop("Output")
        return self

    def convert_to_pcs(self):
        pca = PCA(n_components=len(self.df.columns))
        principalComponents = pca.fit_transform(self.df)
        pc_column_names = ["PC" + str(i)
                           for i in range(1, len(self.df.columns)+1)]
        self.df = pd.DataFrame(data=principalComponents,
                               columns=pc_column_names)
        return self

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

    def standardize(self):
        scaled_features = StandardScaler().fit_transform(self.df.values)
        self.df = pd.DataFrame(
            scaled_features, index=self.df.index, columns=self.df.columns)
        return self

    def normalize(self):
        scaled_features = MinMaxScaler().fit_transform(self.df.values)
        self.df = pd.DataFrame(
            scaled_features, index=self.df.index, columns=self.df.columns)
        return self

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


def default():
    return Preprocess(
        "data/full.csv").code_output().drop_output().onehot_categorical()


def pinpoint_pcs():
    data = default()

    data.standardize()
    data.convert_to_pcs()
    data.remove_low_impact(.25)
    data.drop_outliers(3)

    return data


if __name__ == "__main__":

    data = default()

    data.normalize()
    data = data.convert_to_pcs()
    data.graph_3d_pca()
