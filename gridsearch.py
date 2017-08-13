from sklearn.model_selection import GridSearchCV


def grid_search(xgb_regressor, param_test, x_train, y_train):
    gsearch = GridSearchCV(
        xgb_regressor,
        param_grid=param_test,
        n_jobs=8)
    
    print "start grid search ..."
    gsearch.fit(x_train, y_train)

    print "cv_results_ \n{0}\n\n".format(gsearch.cv_results_)
    print "best_params_ \n{0}\n\n".format(gsearch.best_params_)
    print "best_score_ \n{0}\n\n".format(gsearch.best_score_)