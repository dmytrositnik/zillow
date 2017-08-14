import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from modelfit import modelfit
from traintopred import train_to_pred
from gridsearch import grid_search

properties = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\properties_2016.csv")
train = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\train_2016_v2.csv")
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.46388]
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

xgb_regressor = XGBRegressor(
    learning_rate=0.06,
    max_depth=5,
    subsample=0.73,
    objective='reg:linear',
    eval_metric='mae',
    base_score=y_mean,
    gamma=0,
    silent=True)

modelfit(xgb_regressor, x_train, y_train)

xgb_regressor.fit(x_train, y_train)
pred = xgb_regressor.predict(x_test)

y_pred = []

for i, predict in enumerate(pred):
    y_pred.append(str(round(predict, 4)))
y_pred = np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_pred, '201611': y_pred, '201612': y_pred,
                       '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
