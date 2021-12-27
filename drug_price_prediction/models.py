import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from .utils.models_utils import mape_error

mape = make_scorer(mape_error, greater_is_better=False)

def data_preparation(train_df, keep_features, test_size=0.2, target_col='logprice', random_state=None):
    """Permute and split DataFrame index into train and test.
    Parameters
    ----------
    train_df: pandas.DataFrame
    test_size: float
        Fraction between 0.0 and 1.0
    random_state: int
    Returns
    -------
    tuple of numpy.ndarray,
        X_train, X_test, y_train, y_test
    """

    logging.info("Splitting the data-frame into train and test parts")
    X = train_df[keep_features].values
    y = train_df[target_col].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def fit_cv(train_df, keep_features, model='RF', n_estimators=20):
    """Fits a regressor on the data using a 5-fold cross validation.
    Parameters
    ----------
    train_df: pandas.DataFrame
    keep_features: List[str]
        List of features to keep in the training set
    model: str
        Model to train ('RF'=Random Forest and 'XG'= XGBoost)
    Returns
    -------
            trained regressor and predictions on the test set
    """
    if model == 'RF':
        hyperparameters_rf = {'n_estimators': n_estimators, 
                'n_jobs': -1, 
                'verbose': 1,
                'max_features': None,
                'min_samples_leaf': 1   
                }

    logging.info("Data preparation")
    X, X_test, y, y_test = data_preparation(train_df, keep_features)

    mape_err = []
    rmse_err = []
    mae_err = []

    kfold = KFold(n_splits=5)
    kfold = KFold(n_splits=5)
    kfold.get_n_splits(X)

    for train_index, valid_index in kfold.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model == 'RF':
            regressor = RandomForestRegressor(**hyperparameters_rf)
        t0 = time.time()
        regressor.fit(X_train, y_train)
        logging.info("Fit in %0.3fs" % (time.time() - t0))
        y_pred = regressor.predict(X_valid)

        curr_mape_err = mape_error(np.exp(y_valid), np.exp(y_pred))
        curr_rmse_err = np.sqrt(mean_squared_error(np.exp(y_valid), np.exp(y_pred)))
        curr_mae_err = mean_absolute_error(np.exp(y_valid), np.exp(y_pred))
        
        mape_err.append(curr_mape_err)
        rmse_err.append(curr_rmse_err)
        mae_err.append(curr_mae_err)
        print("MAPE error: {} | MSE error: {} | MAE error: {}".format(curr_mape_err, curr_rmse_err, curr_mae_err))
            
    logging.info("=== MAPE Error : mean = {} ; std = {} ===".format(np.mean(mape_err), np.std(mape_err)))
    logging.info("=== RMSE Error : mean = {} ; std = {} ===".format(np.mean(rmse_err), np.std(rmse_err)))
    logging.info("=== MAE Error : mean = {} ; std = {} ===".format(np.mean(mae_err), np.std(mae_err)))

    logging.info("Inference on test set")
    y_pred_test = regressor.predict(X_test)

    r = pearsonr(y_test, y_pred_test)
    mape_test = mape_error(np.exp(y_test), np.exp(y_pred_test))
    rmse_test = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test)))
    mae_test = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))

    logging.info("===On the test set: MAPE={} | RMSE={} | MAE={} | Pearson (Log) r = {}===".format(mape_test, rmse_test, mae_test, r))

    return regressor, mape_test
def fit_cv_random_search(train_df, keep_features, model='RF'):
    """Fits a regressor on the data using a 5-fold cross validation.
    Parameters
    ----------
    train_df: pandas.DataFrame
    keep_features: List[str]
        List of features to keep in the training set
    model: str
        Model to train ('RF'=Random Forest and 'XG'= XGBoost)
    Returns
    -------
            trained regressor and predictions on the test set
    """
    if model == 'RF':
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
              'max_features': ['auto', 'sqrt']
              }
    logging.info("Data preparation")
    X_train, X_test, y_train, y_test = data_preparation(train_df, keep_features)

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 2,
                                cv = 3,
                                verbose=2,
                                random_state=42,
                                n_jobs = -1
                                )
    rf_random.fit(X_train, y_train)

    logging.info("Best parameters {} for {} model".format(model, rf_random.best_params_))
    best_model = rf_random.best_estimator_
    y_pred_test = best_model.predict(X_test)

    logging.info("Running inference with best {} model".format(model))
    r = pearsonr(y_test, y_pred_test)
    mape_test = mape_error(np.exp(y_test), np.exp(y_pred_test))
    rmse_test = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test)))
    mae_test = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))

    logging.info("===On the test set: MAPE={} | RMSE={} | MAE={} | Pearson (Log) r = {}===".format(mape_test, rmse_test, mae_test, r))

    return best_model


