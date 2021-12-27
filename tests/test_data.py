import pandas as pd
from drug_price_prediction.data import encode_binary, normalize_text, add_log_target, transform_date_features
from drug_price_prediction.data import encode_ordinal, one_hot_encode_categorical
from pandas.util.testing import assert_frame_equal
import numpy as np 

def test_normalize_text():
    test_dict = {'drug_id': ['0'],
                'description': ['Orally aCtiVe']
                }
    
    expected_dict = {'drug_id': ['0'],
                 'description': ['orally active']
                }

    result = normalize_text(pd.DataFrame.from_dict(test_dict))
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)

def test_add_log_target():
    test_dict = {'drug_id': ['0'],
                'price': [1.45]
                }
    
    expected_dict = {'drug_id': ['0'],
                'price': [1.45],
                'logprice': [np.log(1.45)]
                }
    result = add_log_target(pd.DataFrame.from_dict(test_dict), 'price')
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)

def test_transform_date_features():
    test_dict = {'drug_id': ['0'],
        'date': ['20140101']
        }
    
    expected_dict = {'drug_id': ['0'],
                 'date': [2014]
                }

    result = transform_date_features(pd.DataFrame.from_dict(test_dict), ['date'])
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)

def test_encode_binary():
    test_dict = {'drug_id': ['0', '1', '2'],
        'binary': ['Yes', 'No', 'Yes']
        }
    
    expected_dict = {'drug_id': ['0', '1', '2'],
        'binary': [1, 0, 1]
        }

    result = encode_binary(pd.DataFrame.from_dict(test_dict), {'Yes': 1, 'No': 0}, ['binary'])
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)

def test_encode_ordinal():
    test_dict = {'drug_id': ['0', '1', '2', '3', '4', '5', '6'],
        'ordinal': ['0%', '10%', '20%', '10%', '0%', '10%', '20%']
        }
    
    expected_dict = {'drug_id': ['0', '1', '2', '3', '4', '5', '6'],
        'ordinal': [0, 1, 2, 1, 0, 1, 2]
        }

    result = encode_ordinal(pd.DataFrame.from_dict(test_dict), ['ordinal'])
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)

def test_encode_categorical():
    test_dict = {'drug_id': ['0', '1', '2', '3', '4', '5', '6'],
        'categorical': ['approved_1', 'approved_2', 'approved_3', 'approved_1', 'approved_2', 'approved_3', 'approved_1']
    }
    
    expected_dict = {'drug_id': ['0', '1', '2', '3', '4', '5', '6'],
        'categorical=approved_1': [1, 0, 0, 1, 0, 0, 1],
        'categorical=approved_2': [0, 1, 0, 0, 1, 0, 0],
        'categorical=approved_3': [0, 0, 1, 0, 0, 1, 0]
        }

    result = one_hot_encode_categorical(pd.DataFrame.from_dict(test_dict), ['categorical'])
    expected = pd.DataFrame.from_dict(expected_dict)

    assert_frame_equal(result, expected)












