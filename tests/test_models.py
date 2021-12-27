from drug_price_prediction.data import normalize_text, add_log_target, encode_binary, encode_ordinal
from drug_price_prediction.data import one_hot_encode_categorical, encode_text, transform_date_features
from drug_price_prediction.models import data_preparation, fit_cv
from drug_price_prediction.constants import text_features, feat_categorical, feat_binary, feat_ordinal, feat_dates
import pandas as pd
import os

validation_data_path = os.path.join(os.path.dirname(__file__), "validation_data/drugs_validation.csv")

def test_run_model():
    # Import and merge data
    validation_df = pd.read_csv(validation_data_path, index_col=0)
    active_ingredients = pd.read_csv(os.path.join(os.path.dirname(__file__), "validation_data/active_ingredients.csv"))
    valid_index = validation_df['drug_id'].values
    dict_ingredients_valid = {k: active_ingredients[active_ingredients.drug_id==k]['active_ingredient'].values[0] for k in valid_index}
    validation_df['active_ingredient'] = validation_df['drug_id'].map(dict_ingredients_valid)

    # Pre-processing pipeline
    processed_df = normalize_text(validation_df)
    processed_df = add_log_target(processed_df, 'price')
    processed_df = transform_date_features(processed_df, feat_dates)
    processed_df = encode_binary(processed_df, feat_binary)
    processed_df = encode_ordinal(processed_df, feat_ordinal)
    processed_df = one_hot_encode_categorical(processed_df, feat_categorical)
    processed_df = encode_text(processed_df, text_features)

    # Fit model
    KEEP_FEATURES = [col for col in processed_df.columns if col not in ['price', 'logprice', 'drug_id']]

    model, mape_score, _, _ = fit_cv(processed_df, KEEP_FEATURES, model='RF', n_estimators=20)
    assert(mape_score < 100)






