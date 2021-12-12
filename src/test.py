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
    "data/training_stratified_80.csv").code_output().drop_output()

ppc.print_correlation_pairs()
plt.matshow(ppc.df.corr())
plt.show()
