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
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier


# load data set from csv (converted from online xlsx)
df = pd.read_csv('BankChurners.csv')
df = df.sample(frac=1, random_state=0)  # shuffle

# fetch the "Y" column (output)
df["Attrition_Flag"] = df["Attrition_Flag"].apply(
    lambda x: 1 if x == "Attrited Customer" else 0)

# one-hot categorical variables
df = pd.get_dummies(
    df, columns=['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])

# get output
y = df["Attrition_Flag"].values

# get x column
X = df.loc[:, 'Customer_Age':]

i = 0
for feature in X.columns:
    print("feature ", i, ": ", feature)
    i += 1

# normalize (motly for visualization)
X = StandardScaler().fit_transform(X)

"""
# compute pca
pca = PCA(n_components=37)
principalComponents = pca.fit_transform(X)
X = pd.DataFrame(data=principalComponents)
"""

"""
my_features = X.columns

for feature1 in my_features:
    for feature2 in my_features:
        if feature1 != feature2:
            X[feature1 + "_div_" + feature2] = X[feature1] / (X[feature2] + 1)
"""

clf = XGBClassifier(n_estimators=10, eval_metric='logloss',
                    use_label_encoder=False)

scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
print(scores)
print("Average: ", sum(scores) / len(scores))

clf.fit(X, y)

print("Training accuracy: ", accuracy_score(
    y, clf.predict(X)))

plot_confusion_matrix(clf, X, y, values_format='d')
plt.show()

"""
# get importance
importance = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
feature_names = [f"feature {i}" for i in range(X.shape[1])]
importances = pd.Series(importance.importances_mean, index=feature_names).sort_values(ascending=False)
fig, ax = plt.subplots()
importances.plot.bar(yerr=importance.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
"""

"""
# compute pca
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=[
    'principal component 1', 'principal component 2', 'principal component 3'])

# plot pc1, pc2, and pc3 axes
fig = plt.figure()
ax = plt.axes(projection='3d')

# labels
ax.set_xlabel('Principal Component 1', fontsize=8)
ax.set_ylabel('Principal Component 2', fontsize=8)
ax.set_zlabel('Principal Component 3', fontsize=8)
ax.set_title('3 component PCA', fontsize=20)
ax.scatter3D(principalDf['principal component 1'],
             principalDf['principal component 2'],
             principalDf['principal component 3'],
             c=y, cmap='Blues')

# for i in range(len(X)):
#    ax.annotate(i, (principalDf['principal component 1'][i],
#                    principalDf['principal component 2'][i]))

ax.grid()
plt.show()
"""
