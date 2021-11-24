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
from preprocess import Preprocess

# 5K-CV AUC-mu 99.31
# Test AUC 94.10

# load data set from csv (converted from online xlsx)
train_ppc = Preprocess(
    "../data/training_stratified_80.csv").code_output().drop_output().onehot_categorical()

train_X = train_ppc.df
train_y = train_ppc.output

clf = XGBClassifier(n_estimators=60, eval_metric='logloss', scale_pos_weight=5.2,
                    use_label_encoder=False)

print("Cross-validation 5-fold AUC scores:")
scores = cross_val_score(clf, train_X, train_y, cv=5, scoring='roc_auc')
print(scores)
print("Average: ", sum(scores) / len(scores))

# Test set
clf.fit(train_X, train_y)

test_ppc = Preprocess(
    "../data/testing_stratified_20.csv").code_output().drop_output().onehot_categorical()

test_X = test_ppc.df.to_numpy()
test_y = test_ppc.output.to_numpy()

print("Testing AUC: ", metrics.roc_auc_score(
    test_y, clf.predict(test_X)))

plot_confusion_matrix(clf, test_X, test_y, values_format='d')
plt.show()

"""
importance = permutation_importance(
    clf, train_X, train_y, n_repeats=10, random_state=0)
feature_names = [f"feature {i}" for i in range(train_X.shape[1])]
importances = pd.Series(importance.importances_mean,
                        index=feature_names).sort_values(ascending=False)
fig, ax = plt.subplots()
importances.plot.bar(yerr=importance.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
"""
