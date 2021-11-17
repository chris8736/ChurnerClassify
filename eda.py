import numpy as np
import matplotlib.pyplot as plt  # visualize PCA
import pandas as pd  # import data from file
import math
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier


# load data set from csv (converted from online xlsx)
df = pd.read_csv('BankChurners.csv')

print(df.info())
df = df.sample(frac=1, random_state=0)  # shuffle

# fetch the "Y" column (output)
df["Attrition_Flag"] = df["Attrition_Flag"].apply(
    lambda x: 1 if x == "Attrited Customer" else 0)

"""
# one-hot categorical variables
df = pd.get_dummies(
    df, columns=['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
"""

# get output
y = df["Attrition_Flag"].values

# get x column
X = df.loc[:, 'Customer_Age':]

for feature in X.columns:
    sns_plot = seaborn.displot(df[feature])
    plt.show()

# normalize (motly for visualization)
X = StandardScaler().fit_transform(X)
