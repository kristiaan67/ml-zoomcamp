import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_file = '../data/airbnb_data.csv'
feature_set = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'room_type', 'price']

df = pd.read_csv(data_file, usecols=feature_set)
df = df.fillna(0)
df['price'] = np.log1p(df['price'])
df.head()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

del df_train['price']
del df_val['price']
del df_test['price']

dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
print(export_text(dt, feature_names=dv.get_feature_names()))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("rmse", "=", rmse)

rmses = []
for estimators in range(10, 200, 10):
    rf = RandomForestRegressor(n_estimators=estimators, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmses.append((estimators, rmse))
    
df_rmses = pd.DataFrame(rmses, columns=["n_estimators", "rmse"])
print(df_rmses)

#plt.plot(df_rmses.n_estimators, df_rmses.rmse)


rmses = []
for depth in [10, 15, 20, 25]:
    for estimators in range(10, 200, 10):
        rf = RandomForestRegressor(max_depth=depth, n_estimators=estimators, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmses.append((depth, estimators, rmse))
        
df_rmses = pd.DataFrame(rmses, columns=["depth", "n_estimators", "rmse"])
print(df_rmses)


#for depth in [10, 15, 20, 25]:
#    df_subset = df_rmses[df_rmses.depth == depth]
#    
#    plt.plot(df_subset.n_estimators, df_subset.rmse,
#             label='max_depth=%d' % depth)
#plt.legend()


rf = RandomForestRegressor(max_depth=20, n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
res = pd.DataFrame(dv.get_feature_names(), rf.feature_importances_)
print(res)

import xgboost as xgb

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

rmses = []
for eta in [0.3, 0.1, 0.01]:
    xgb_params = {
        'eta': eta, 
        'max_depth': 6,
        'min_child_weight': 1,

        'objective': 'reg:squarederror',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    y_pred = model.predict(dval)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmses.append((eta, rmse))
    print("done", eta)
    
df_rmses = pd.DataFrame(rmses, columns=["eta", "rmse"])
print(df_rmses)

