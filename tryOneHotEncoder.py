# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:42:19 2019

@author: 
"""

# Import scikit-learn dataset library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import data_preparation  # importing Data_preperation module
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
# To install category_encoders package
# conda install -c conda-forge category_encoders
import category_encoders as ce

# Load dataset
data = pd.read_csv( 'something')
# Features

input_columns = data.columns.drop(['A', 'B', 'C'])
X = data[input_columns]
y = data['classification']  # Labels

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=5)
# ////////////////
# OneHotEncoder
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)
# ////////////////

# To-Do: Add a piece of code to select best algorithm

# Create a Gaussian Classifier (Example)
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
##clf.fit(X_train, y_train)
clf.fit(X_train_ohe, y_train)  # ///////

# #y_pred = clf.predict(X_test)
y_pred = clf.predict(X_test_ohe)  # ///////

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# #res = X_test.copy()
res = X_test_ohe.copy()  # ///////
res['real'] = y_test
# Get the actual predictions on test data
res['prediction'] = y_pred
res['result'] = np.where(res['real'] == res['prediction'], 'model was RIGHT',
                         'model was WRONG')

res.to_excel('result_random_forest.xlsx')

# #index = [input_columns]
index = X_train_ohe.columns[:]  # ///////
feature_imp = pd.Series(clf.feature_importances_,
                        index=index).sort_values(ascending=False)

# Creating a bar plot
sns.barplot(x=feature_imp.head(10), y=feature_imp.head(10).index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.tight_layout()
plt.show()
