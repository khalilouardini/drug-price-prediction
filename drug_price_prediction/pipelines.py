import logging
import pandas as pd
import numpy as np
import os
from drug_price_prediction import data, models
from drug_price_prediction.constants import text_features, feat_categorical, feat_binary, feat_ordinal, feat_dates

def run_drug_price_prediction(data_dir, model, n_estimators):
    """Data pipeline and predictions on train set.
    Parameters
    ----------
    data_dir: str
        Path to the data directory
    mdoel: str
        Model to train (Random Forest ('RF') or XGBoost ('XG'))
    n_estimators: int
        Number of estimators in ensembling model
    """
    logging.info('Starting the pipeline')

    # Import and merge train data
    train_df = pd.read_csv(data_dir + 'drugs_train.csv')
    active_ingredients = pd.read_csv(data_dir + 'active_ingredients.csv')
    train_index = train_df['drug_id'].values
    dict_ingredients_train = {k: active_ingredients[active_ingredients.drug_id==k]['active_ingredient'].values[0] for k in train_index}
    train_df['active_ingredient'] = train_df['drug_id'].map(dict_ingredients_train)

    # Pre-processing pipeline for train set
    processed_df = data.normalize_text(train_df)
    processed_df = data.add_log_target(processed_df, 'price')
    processed_df = data.transform_date_features(processed_df, feat_dates)
    processed_df = data.encode_binary(processed_df, feat_binary)
    processed_df = data.encode_ordinal(processed_df, feat_ordinal)
    processed_df = data.one_hot_encode_categorical(processed_df, feat_categorical)
    processed_df = data.encode_text(processed_df, text_features)

    # Fit model
    KEEP_FEATURES = [col for col in processed_df.columns if col not in ['price', 'logprice', 'drug_id']]
    model, mape_score, mse_score, mae_score = models.fit_cv(processed_df,
                                                            KEEP_FEATURES,
                                                            model=model,
                                                            n_estimators=n_estimators
                                                            )

    return model, mape_score, mse_score, mae_score

def run_inference(data_dir):
    """Data pipeline and predictions on new unseen data.
    Parameters
    ----------
    data_dir: str
        Path to the data directory
    """
    logging.info("Fitting model on the train set...")
    model, _, _, _ = run_drug_price_prediction(data_dir) 

    # Import and merge train data
    test_df = pd.read_csv(data_dir + 'drugs_test.csv')
    active_ingredients = pd.read_csv(data_dir + 'active_ingredients.csv')
    test_index = test_df['drug_id'].values
    dict_ingredients_test = {k: active_ingredients[active_ingredients.drug_id==k]['active_ingredient'].values[0] for k in test_index}
    test_df['active_ingredient'] = test_df['drug_id'].map(dict_ingredients_test)

    # Pre-processing pipeline for test set
    processed_df = data.normalize_text(test_df)
    processed_df = data.add_log_target(processed_df, 'price')
    processed_df = data.transform_date_features(processed_df, feat_dates)
    processed_df = data.encode_binary(processed_df, feat_binary)
    processed_df = data.encode_ordinal(processed_df, feat_ordinal)
    processed_df = data.one_hot_encode_categorical(processed_df, feat_categorical)
    processed_df = data.encode_text(processed_df, text_features)

    logging.info("Inference on test set...")
    # Inference
    KEEP_FEATURES = [col for col in processed_df.columns if col not in ['drug_id']]
    X_test = processed_df[KEEP_FEATURES].values

    y_pred = np.exp(model.predict(X_test))
    submissions_dict = {'drug_id': test_index, 'price': y_pred}

    if not os.path.exists('../results'):
        os.mkdir('../results')

    pd.DataFrame.from_dict(submissions_dict).to_csv('../results/' + 'predictions.csv')
