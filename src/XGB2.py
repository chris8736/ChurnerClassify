import numpy as np
import matplotlib.pyplot as plt  # visualize PCA
import pandas as pd  # import data from file
import math
import statistics
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from preprocess import Preprocess
from sklearn.model_selection import KFold

# 5K-CV AUC-mu 99.28 +- .32
# Test AUC 95.54

# load data set from csv (converted from online xlsx)
train_ppc = Preprocess(
    "../data/training_stratified_80.csv").code_output().drop_output().onehot_categorical()

train_ppc.remove_columns(['Dependent_count', 'Months_on_book', 'Avg_Open_To_Buy', 'Education_Level_College', 'Education_Level_Graduate', 'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
                          'Marital_Status_Divorced', 'Income_Category_$120K +', 'Income_Category_$80K - $120K', 'Card_Category_Blue', 'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'])

train_X = train_ppc.df
train_y = train_ppc.output

clf = XGBClassifier(n_estimators=250, eval_metric='logloss', scale_pos_weight=5.2, learning_rate=0.26, max_depth=2, gamma=0, subsample=1, colsample_bytree=1,
                    use_label_encoder=False)

# param_grid = {'n_estimators': [255, 260, 265],
#              'learning_rate': [.258, .26, .262]}
#clf = GridSearchCV(xgb, param_grid, scoring='roc_auc')
#clf.fit(train_X, train_y)
# print(clf.best_params_)

scores = []
for i in range(3):
    kf = KFold(shuffle=True)
    print("Cross-validation 5-fold AUC scores:")
    scores = scores + list(cross_val_score(
        clf, train_X, train_y, cv=kf, scoring='roc_auc'))
print(scores)
print("Average: ", sum(scores) / len(scores))
print("Stdev: ", statistics.stdev(scores))

# Test set

test_ppc = Preprocess(
    "../data/testing_stratified_20.csv").code_output().drop_output().onehot_categorical()

test_ppc.remove_columns(['Dependent_count', 'Months_on_book', 'Avg_Open_To_Buy', 'Education_Level_College', 'Education_Level_Graduate', 'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
                         'Marital_Status_Divorced', 'Income_Category_$120K +', 'Income_Category_$80K - $120K', 'Card_Category_Blue', 'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'])

clf.fit(train_X, train_y)

test_X = test_ppc.df.to_numpy()
test_y = test_ppc.output.to_numpy()

print("Testing AUC: ", metrics.roc_auc_score(
    test_y, clf.predict(test_X)))

plot_confusion_matrix(clf, test_X, test_y, values_format='d')
plt.show()
