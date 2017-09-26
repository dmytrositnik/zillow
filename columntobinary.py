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

def column_to_binary(properties, columns):
    return pd.get_dummies(properties, columns=columns)





