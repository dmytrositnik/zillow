import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from modelfit import modelfit

properties = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\properties_2016.csv")
train = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\train_2016_v2.csv", parse_dates=["transactiondate"])
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')

# add transaction month column
train_df['transaction_month'] = train_df['transactiondate'].dt.month

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

modelfit(xgb_regressor, x_train, y_train, show_feature_importance=True)

xgb_regressor.fit(x_train, y_train)

# add month 201610
x_test['transaction_month'] = 10

pred = xgb_regressor.predict(x_test)

y_pred201610 = []

for i, predict in enumerate(pred):
    y_pred201610.append(str(round(predict, 4)))
y_pred201610 = np.array(y_pred201610)

# add month 201611
x_test['transaction_month'] = 11

pred = xgb_regressor.predict(x_test)

y_pred201611 = []

for i, predict in enumerate(pred):
    y_pred201611.append(str(round(predict, 4)))
y_pred201611 = np.array(y_pred201611)

# add month 201612
x_test['transaction_month'] = 12

pred = xgb_regressor.predict(x_test)

y_pred201612 = []

for i, predict in enumerate(pred):
    y_pred201612.append(str(round(predict, 4)))
y_pred201612 = np.array(y_pred201612)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_pred201610, '201611': y_pred201611, '201612': y_pred201612,
                       '201710': y_pred201610, '201711': y_pred201611, '201712': y_pred201612})
# set col 'ParcelID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
