import pandas as pd
from drug_price_prediction import data

def test_normalize_text():
    test_dict = {'drug_1': 'oràlly àCtiVe',
                 'drug_2': 'Par voie cutannéE',
                 'drug_3': 'PLaquettes thermoformées'
                  }
    
    expected_dict = {'drug_1': 'orally active',
                 'drug_2': 'par voie cutannee',
                 'drug_3': 'plaquettes thermoformees'
                  }

    df = pd.DataFrame.from_dict(test_dict)



