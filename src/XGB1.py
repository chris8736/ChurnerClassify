import numpy as np
import matplotlib.pyplot as plt  # visualize PCA
import pandas as pd  # import data from file
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier
from preprocess import Preprocess

# AUC 99.29

# load data set from csv (converted from online xlsx)
ppc = Preprocess(
    "../data/training_stratified_80.csv").code_output().drop_output().onehot_categorical()

X = ppc.df.to_numpy()
y = ppc.output.to_numpy()

clf = XGBClassifier(n_estimators=60, eval_metric='logloss',
                    use_label_encoder=False)

scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(scores)
print("Average: ", sum(scores) / len(scores))

clf = XGBClassifier(n_estimators=60, eval_metric='logloss',
                    use_label_encoder=False)
clf.fit(X, y)

# Test set
test_ppc = Preprocess(
    "../data/testing_stratified_20.csv").code_output().drop_output().onehot_categorical()

test_X = test_ppc.df.to_numpy()
test_y = test_ppc.output.to_numpy()

print("Testing accuracy: ", metrics.accuracy_score(
    test_y, clf.predict(test_X)))

plot_confusion_matrix(clf, test_X, test_y, values_format='d')
plt.show()
