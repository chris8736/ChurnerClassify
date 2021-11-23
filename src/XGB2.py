import numpy as np
import matplotlib.pyplot as plt  # visualize PCA
import pandas as pd  # import data from file
import math
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from preprocess import Preprocess

# 5K-CV AUC-mu 99.34
# Test AUC 94.22

# load data set from csv (converted from online xlsx)
train_ppc = Preprocess(
    "../data/training_stratified_80.csv").code_output().drop_output().onehot_categorical()

train_X = train_ppc.df
train_y = train_ppc.output

xgb = XGBClassifier(n_estimators=60, eval_metric='logloss', scale_pos_weight='5.2', learning_rate=0.3, reg_lambda=0,
                    use_label_encoder=False)

param_grid = {'n_estimators': [102]}
clf = GridSearchCV(xgb, param_grid, scoring='roc_auc')
clf.fit(train_X, train_y)
print(clf.best_params_)

print("Cross-validation 5-fold AUC scores:")
scores = cross_val_score(clf, train_X, train_y, cv=5, scoring='roc_auc')
print(scores)
print("Average: ", sum(scores) / len(scores))

# Test set

test_ppc = Preprocess(
    "../data/testing_stratified_20.csv").code_output().drop_output().onehot_categorical()

clf.fit(train_X, train_y)

test_X = test_ppc.df.to_numpy()
test_y = test_ppc.output.to_numpy()

print("Testing AUC: ", metrics.roc_auc_score(
    test_y, clf.predict(test_X)))

plot_confusion_matrix(clf, test_X, test_y, values_format='d')
plt.show()
