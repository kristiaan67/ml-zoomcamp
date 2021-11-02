#!/usr/bin/env python
# coding: utf-8


## Imports

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import pickle


## Global Variables

data_file = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
random_state = 1234
random.seed(random_state)


## Load Data Set
df = pd.read_csv(data_file)
df.columns = df.columns.str.lower()


## Data Preparation

# 1. Remove the employee number from the model since unique for each employee and the employee coount/over 18/ standard hours features since always 1/Y/80. Those features are not relevant for the classification model.

del df['employeenumber']
del df['employeecount']
del df['over18']
del df['standardhours']


# 2. Map numerical values to labels for categorical features.

education_values = {
    0: 'unk',
    1: 'Below College',
    2: 'College',
    3: 'Bachelor',
    4: 'Master',
    5: 'Doctor'
}
df.education = df.education.map(education_values)


environmentsatisfaction_values = {
    0: 'unk',
    1: 'low',
    2: 'medium',
    3: 'high',
    4: 'very high'
}
df.environmentsatisfaction = df.environmentsatisfaction.map(environmentsatisfaction_values)


jobinvolvement_values = {
    0: 'unk',
    1: 'low',
    2: 'medium',
    3: 'high',
    4: 'very high'
}
df.jobinvolvement = df.jobinvolvement.map(jobinvolvement_values)


jobsatisfaction_values = {
    0: 'unk',
    1: 'low',
    2: 'medium',
    3: 'high',
    4: 'very high'
}
df.jobsatisfaction = df.jobsatisfaction.map(jobsatisfaction_values)


performancerating_values = {
    0: 'unk',
    1: 'low',
    2: 'good',
    3: 'excellent',
    4: 'outstanding'
}
df.performancerating = df.performancerating.map(performancerating_values)


relationshipsatisfaction_values = {
    0: 'unk',
    1: 'low',
    2: 'medium',
    3: 'high',
    4: 'very high'
}
df.relationshipsatisfaction = df.relationshipsatisfaction.map(relationshipsatisfaction_values)

worklifebalance_values = {
    0: 'unk',
    1: 'bad',
    2: 'good',
    3: 'better',
    4: 'best'
}
df.worklifebalance = df.worklifebalance.map(worklifebalance_values)


# 3. Convert boolean values to 1 and 0:

df['attrition'] = (df.attrition == 'Yes').astype(int)
df['overtime'] = (df.overtime == 'Yes').astype(int)


# 4. Create training, validation and test data sets:

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=random_state)

y_full_train = df_full_train['attrition']
y_train = df_train['attrition']
y_val = df_val['attrition']
y_test = df_test['attrition']

del df_full_train['attrition']
del df_train['attrition']
del df_val['attrition']
del df_test['attrition']


df_test.to_csv('/output/attrition_test_data.csv', index=False)

## Explorative Data Analysis
 
### Feature Analysis

# 1. Check for missing values:

print("Null values:\n", df_full_train.isnull().sum())


# **We don’t have to handle missing values** in the dataset: all the values in all the columns are present.

# 2. Check the distribution of the target variable (majority vs. minority class)

print("Distribution target variable:\n", y_train.value_counts())

print(round(y_train.mean(), 2) * 100, "percent suffered from attrition, which makes the data set imbalanced.")

# 3. Retrieve the numerical and categorical values

numeric_feats = []
categorical_feats = []
for col in df_full_train.columns:
    if col == 'overtime':
        categorical_feats.append(col) # boolean feature is handled as categorical 
    elif (df_full_train[col].dtype == np.float64 or df_full_train[col].dtype == np.int64):
        numeric_feats.append(col)
    else:
        categorical_feats.append(col)
            


# Create dictionnaries

dv = DictVectorizer(sparse=False)


full_train_dict = df_full_train[categorical_feats + numeric_feats].to_dict(orient='records')
dv.fit(full_train_dict)
X_full_train = dv.transform(full_train_dict)

train_dict = df_train[categorical_feats + numeric_feats].to_dict(orient='records')
X_train = dv.transform(train_dict)

val_dict = df_val[categorical_feats + numeric_feats].to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test[categorical_feats + numeric_feats].to_dict(orient='records')
X_test = dv.transform(test_dict)


## Machine Learning Models

# Store the results of the ML models
test_data_results = []

# we generate an index from Cross Validation with 10 possible folds
split_index = [-1 if x in df_train.index else random.randint(0, 9) for x in df_full_train.index]
pds = PredefinedSplit(test_fold = split_index)

### Some functions used for tuning and validating the models

def validate_results(y_val, y_pred, label=''):
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)
    print('\n#### ' + label)
    print("Accuracy Score: %.3f" % accuracy)
    print("Area Under ROC Curve: %.3f" % roc_auc)
    print("Recall score: %.3f" % recall)
    print(f"Confusion Matrix: \n{conf_matrix}")

    return (accuracy, roc_auc, recall, confusion_matrix)


def create_grid_search_cv(model, hyperparam, scoring="roc_auc"):
    grid = GridSearchCV(model, hyperparam, scoring=scoring, cv=pds, n_jobs=-1, refit=True)
    grid.fit(X_full_train, y_full_train)
    print(f'Best score: {round(grid.best_score_, 3)} with param: {grid.best_params_}')
    return(grid)

def predict(df, model):
    cat = df[categorical_feats + numeric_feats].to_dict(orient='records')
    X = dv.transform(cat)
    y_pred = model.predict(X)
    return y_pred


### Logistic Regression

model_lg = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=random_state)
model_lg.fit(X_train, y_train)
y_pred_lg = model_lg.predict(X_val)
validate_results(y_val, y_pred_lg, "Simple Logistic Regression")


### Weighted Logistic Regression

first_weight = {0: y_train.value_counts()[1], 1: y_train.value_counts()[0]}
model_lg = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=random_state, class_weight=first_weight)
model_lg.fit(X_train, y_train)
y_pred_lg = model_lg.predict(X_val)
validate_results(y_val, y_pred_lg, "Weighted Logistic Regression (base)")


#### Tuning the model

##### Tuning Class Weight

# First we check whether another weight might lead to a further improvement.
w_range = [first_weight, {0:1, 1:4}, {0:1, 1:4.5}, {0:1, 1:5}, {0:1, 1:5.5}, {0:1, 1:6}]
hyperparam_lg = {"class_weight": w_range}

model_lg = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=random_state)
grid_lg = create_grid_search_cv(model_lg, hyperparam_lg)

best_weight = grid_lg.best_params_['class_weight']
print("Best weights: ", best_weight)

y_pred_lg = grid_lg.predict(X_val)
validate_results(y_val, y_pred_lg, "Weighted Logistic Regression (class weight)")


# ##### Tuning C
# Now we try some improve the model by also tuning the parameter **C**:

C_range = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
hyperparam_lg = {"C": C_range}

model_lg = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=random_state, class_weight=best_weight)
grid_lg = create_grid_search_cv(model_lg, hyperparam_lg)

best_C = grid_lg.best_params_['C']
print("Best C:", best_C)

y_pred_lg = grid_lg.predict(X_val)
validate_results(y_val, y_pred_lg, "Weighted Logistic Regression (tuned)")


#### Final Test on the Test Data Set

def train_lg(df, y):
    cat = df[categorical_feats + numeric_feats].to_dict(orient='records')
    X = dv.transform(cat)
    model = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=random_state, class_weight=best_weight, C=best_C)
    model.fit(X, y)
    return model

model_lg = train_lg(df_full_train, y_full_train)
y_pred_test = predict(df_test, model_lg)
accuracy, roc_auc, recall, confusion_matrix = validate_results(y_test, y_pred_test, "Weighted Logistic Regression (test)")

test_data_results.append(("Weighted Logistic Regression", accuracy, roc_auc, recall))


### Decision Trees

model_dt = DecisionTreeClassifier(random_state=random_state)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_val)
validate_results(y_val, y_pred_dt, "Decision Trees (base)")

#### Tuning the model

# We will perform the following tuning:
# * The class weight (due to the imbalance in the data set)
# * The maximal depth of the tree
# * The minimum number of samples required to be at a leaf node.

##### Tuning Class Weight

w_range = [{0:1, 1:4}, {0:1, 1:4.5}, {0:1, 1:5}, {0:1, 1:5.5}, {0:1, 1:6}]
hyperparam_dt = {"class_weight": w_range}

model_dt = DecisionTreeClassifier(random_state=random_state)
grid_dt = create_grid_search_cv(model_dt, hyperparam_dt)

best_weight = grid_dt.best_params_['class_weight']
print("Best weights: ", best_weight)

y_pred_dt = grid_dt.predict(X_val)
validate_results(y_val, y_pred_dt, "Decision Trees (class weight)")

##### Maximum Depth, Minimum Samples Leaf

depth_range = [1, 2, 3, 4, 5, 6, 10, 15, 50]
min_leaf_range = [1, 5, 10, 15, 20, 50, 100, 200]
hyperparam_dt = {"max_depth": depth_range,
                 "min_samples_leaf": min_leaf_range}
model_dt = DecisionTreeClassifier(random_state=random_state, class_weight=best_weight)
grid_dt = create_grid_search_cv(model_dt, hyperparam_dt)

best_depth = grid_dt.best_params_["max_depth"]
best_min_leaf = grid_dt.best_params_["min_samples_leaf"]

print("Best depth: ", best_depth)
print("Best min leaf size: ", best_min_leaf)

y_pred_dt = grid_dt.predict(X_val)
validate_results(y_val, y_pred_dt, "Decision Trees (tuned)")


#### Final Test on the Test Data Set

def train_dt(df, y):
    cat = df[categorical_feats + numeric_feats].to_dict(orient='records')
    X = dv.transform(cat)
    model = DecisionTreeClassifier(random_state=random_state, class_weight=best_weight, max_depth=best_depth, 
                                   min_samples_leaf=best_min_leaf)
    model.fit(X, y)
    return model

model_dt = train_dt(df_full_train, y_full_train)
y_pred_test = predict(df_test, model_dt)
accuracy, roc_auc, recall, confusion_matrix = validate_results(y_test, y_pred_test, "Decision Trees (test)")

test_data_results.append(("Decision Trees", accuracy, roc_auc, recall))


### Random Forest

model_rf = RandomForestClassifier(random_state=random_state)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_val)
validate_results(y_val, y_pred_rf, "Random Forest (base)")

#### Tuning the model

##### Tuning Class Weight
w_range = [{0:1, 1:4}, {0:1, 1:4.5}, {0:1, 1:5}, {0:1, 1:5.5}, {0:1, 1:6}]
hyperparam_rf = {"class_weight": w_range}
model_rf = RandomForestClassifier(random_state=random_state)
grid_rf = create_grid_search_cv(model_rf, hyperparam_rf)

best_weight = grid_rf.best_params_['class_weight']
print("Best weights: ", best_weight)

y_pred_rf = grid_rf.predict(X_val)
validate_results(y_val, y_pred_rf, "Random Forest (class weight)")

##### Number of Estimators, Maximum Depth, Minimum Samples Leaf

estimators_range = [10, 25, 50, 100, 150]
depth_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
min_leaf_range = [3, 5, 10, 15, 20]
hyperparam_rf = {"n_estimators": estimators_range,
                 "max_depth": depth_range,
                 "min_samples_leaf": min_leaf_range}
model_rf = RandomForestClassifier(random_state=random_state, class_weight=best_weight)
grid_rf = create_grid_search_cv(model_rf, hyperparam_rf)

best_n_estimators = grid_rf.best_params_["n_estimators"]
best_depth = grid_rf.best_params_["max_depth"]
best_min_leaf = grid_rf.best_params_["min_samples_leaf"]

print("Best number of estimators: ", best_n_estimators)
print("Best depth: ", best_depth)
print("Best min leaf size: ", best_min_leaf)

y_pred_rf = grid_rf.predict(X_val)
validate_results(y_val, y_pred_rf, "Random Forest (tuned)")


#### Final Test on the Test Data Set

def train_rf(df, y):
    cat = df[categorical_feats + numeric_feats].to_dict(orient='records')
    X = dv.transform(cat)
    model = RandomForestClassifier(random_state=random_state, class_weight=best_weight, n_estimators=best_n_estimators,
                                   max_depth=best_depth, min_samples_leaf=best_min_leaf)
    model.fit(X, y)
    return model

model_rf = train_rf(df_full_train, y_full_train)
y_pred_test = predict(df_test, model_rf)
accuracy, roc_auc, recall, confusion_matrix = validate_results(y_test, y_pred_test, "Random Forest (test)")

test_data_results.append(("Random Forest", accuracy, roc_auc, recall))


### Gradient Boosting

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.feature_names_)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.feature_names_)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dv.feature_names_)

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': random_state
}
model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred_xgb = model_xgb.predict(dval)
validate_results(y_val, y_pred_xgb >= 0.5, "Gradient Boosting (base)")

#### Tuning the model

##### Maximum Depth, Minimum Child Weight
watchlist = [(dtrain, 'train'), (dval, 'val')]
max_depth_range = [3, 4, 6, 10]
for depth in max_depth_range:
    xgb_params = {
        'eta': 0.3,
        'max_depth': depth,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': random_state
    }
    print("Running with max depth %d" % depth)
    model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=150, verbose_eval=10, evals=watchlist)

# The best depth seems to be **3**.
best_depth = 3
print("Best depth: ", best_depth)

xgb_params = {
    'eta': 0.3,
    'max_depth': best_depth,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': random_state
}
model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred_xgb = model_xgb.predict(dval)
validate_results(y_val, y_pred_xgb >= 0.5, "Gradient Boosting (best depth)")


child_weight_range = [1, 10, 20]
for w in child_weight_range:
    xgb_params = {
        'eta': 0.3,
        'max_depth': best_depth,
        'min_child_weight': w,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': random_state
    }
    print("Running with child weight %d" % w)
    model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10, evals=watchlist)


# The best value seems to be **20** after **30** trees (we will generate 100 to be on te safe side).
best_child_weight = 20
num_boost_round = 100

print("Best child weight: ", best_child_weight)
print("Num boost round: ", num_boost_round)

xgb_params = {
    'eta': 0.3,
    'max_depth': best_depth,
    'min_child_weight': best_child_weight,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': random_state
}
model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)
y_pred_xgb = model_xgb.predict(dval)
validate_results(y_val, y_pred_xgb >= 0.5, "Gradient Boosting (tuned)")


#### Final Test on the Test Data Set

model_xgb = xgb.train(xgb_params, dfulltrain, num_boost_round=num_boost_round, verbose_eval=10)

y_pred_test = model_xgb.predict(dtest)
accuracy, roc_auc, recall, confusion_matrix = validate_results(y_test, y_pred_test >= 0.5, "Gradient Boosting (test)")

test_data_results.append(("Gradient Boosting", accuracy, roc_auc, recall))


## Model Comparison

df_results = pd.DataFrame(test_data_results, columns=["Method", "Accuracy", "ROC AUC", "Recall"])
print(df_results)


# The Decision Trees has not the best accuracy or ROC AUC but we consider it as being the best since it predicts the attrition cases the best which is the aim of this project.

model_best = model_dt
y_pred_best = model_best.predict(X_test)
print("Final results")
validate_results(y_test, y_pred_best, "Final results")


# Save the model and the vectorizer.
model_file = '/output/attrition-model.bin'

with open(model_file, 'wb') as f_out:
    pickle.dump((dv, model_best), f_out)
print("Done. Machine Learning Model saved to '%s'" % model_file)
