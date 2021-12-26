import logging
import unidecode
import unicodedata
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

def normalize_text(df):
    """Normalizes all text in the DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing text data
    Returns
    -------
    pandas.DataFrame
        Data-frame with normalized text
    """

    logging.info("Normalizing text")

    for i in list(df.select_dtypes(include=['object'])):
        #convert all strings to lowercase and remove all accents
        df[i] = df[i].str.lower().map(lambda x: unicodedata.normalize('NFKD', x))
        #removes space if it is first character
        df[i] = df[i].apply(lambda x : x[1:] if x[0]==' ' else x) 
    
    return df

def add_log_target(df, target_col):
    """Adds a new column with the log transformation of a target variable.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing the target variable
    target_col : str 
        name of the target column
    Returns
    -------
    pandas.DataFrame
        Data-frame with additional column 
    """
    df['log' + target_col] = df['price'].apply(np.log)
    return df

def transform_date_features(df, list_features):
    """Transforms date features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing date features
    list features : List[str] 
        list of date features to transform
    Returns
    -------
    pandas.DataFrame
        Data-frame with additional column 
    """
    for feat in list_features:
        df[feat] = df[feat].apply(lambda x: str(x)[:4]).astype(int)
        df[feat] = df[feat].apply(lambda x: str(x)[:4]).astype(int)
    return df

def encode_binary(df, replace_dict, list_features):
    """One hot encoding of binary features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame binary features
    list features : List[str] 
        list of binary features to transform
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    for feat in list_features:
        df.loc[:, feat] = df.loc[:, feat].replace(replace_dict)
        df.loc[:, feat] = df.loc[:, feat].replace(replace_dict)
    return df

def encode_ordinal(df, list_features):
    """Encoding of ordinal features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame binary features
    list features : List[str] 
        list of ordinal features to transform
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    for feat in list_features:
        unique = df['feat'].unique()
        replace_dict = {i: j for i, j in zip(unique, range(len(unique), 0, -1))}
        df.loc[:, feat] = df.loc[:, feat].replace(replace_dict)
    return df

def one_hot_encode_categorical(df, list_features):
    """One hot encoding of categorical features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame binary features
    list features : List[str] 
        list of categorical features to transform
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    vectorizer = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in list_features)
    vectorizer.fit(df[list_features].apply(mkdict, axis=1))  

    categorical_df = pd.DataFrame(vectorizer.transform(df[list_features].apply(mkdict, axis=1)).toarray()).astype(int)
    categorical_df.columns = vectorizer.get_feature_names()
    categorical_df.index = df.index

    df = df.drop(list_features, axis=1)
    df = df.join(categorical_df)

    return df


