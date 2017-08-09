import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

print "start ..."

properties = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\properties_2016.csv")
train_raw = pd.read_csv(r"C:\Users\dmysit\OneDrive\zillow\input\train_2016_v2.csv")
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train = train_raw.merge(properties, how='left', on='parcelid')
train = train.drop(['transactiondate'], axis=1)

target = 'logerror'
IDcol = 'parcelid'


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[IDcol], eval_metric='mae')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[IDcol].values, dtrain_predictions)

    fscocre = alg.get_booster().get_fscore()

    feat_imp = pd.Series(fscocre).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')


"""
# xgboost params
xgb_params = {
    'eta': 0.06,
    'max_depth': 5,
    'subsample': 0.77,
    # 'subsample': 1,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

"""

# Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:linear',
    n_jobs=4,
    scale_pos_weight=1,
    random_state=27)

## modelfit(xgb1, train, predictors)

param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
                                               min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                               objective='reg:linear', n_jobs=4, scale_pos_weight=1, random_state=27),
                        param_grid=param_test1, n_jobs=1, iid=False, cv=5)

print "start grid search ..."
gsearch1.fit(train[predictors], train[target])

print "cv_results_ \n{0}\n\n".format(gsearch1.cv_results_)
print "best_params_ \n{0}\n\n".format(gsearch1.best_params_)
print "best_score_ \n{0}\n\n".format(gsearch1.best_score_)
