import pandas as pd
 

def test_normalize_text():
    test_dict = {'drug_1': 'oràlly àCtiVe',
                 'drug_2': 'Par voie cutannéE',
                 'drug_3': 'PLaquettes thermoformées'
                  }
    df = pd.DataFrame.from_dict(test_dict)
