import pandas as pd
import xgboost as xgb


def modelfit(
        xgb_regressor,
        xtrain,
        ytrain,
        use_train_c_v=True,
        nfold=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=10
):
    if use_train_c_v:
        xgb_param = xgb_regressor.get_xgb_params()
        xgtrain = xgb.DMatrix(xtrain, ytrain)
        cvresult = xgb.cv(
            xgb_param,
            xgtrain,
            nfold=nfold,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval)

        xgb_regressor.set_params(n_estimators=cvresult.shape[0])

        print(len(cvresult))

    # Fit the algorithm on the data
    xgb_regressor.fit(xtrain, ytrain, eval_metric='mae')

    fscocre = xgb_regressor.get_booster().get_fscore()

    feat_imp = pd.Series(fscocre).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance')
