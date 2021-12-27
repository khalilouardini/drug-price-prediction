import re
import unidecode
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def preprocess_text(text):
    """Simple text preprocessing script.
    Parameters
    ----------
    text : str

    Returns
    -------
    str
        processed text
    """
    # 1. Remove punctuation and accetns from text
    text = ''.join([unidecode.unidecode(word) for word in text if word not in punctuation])
    
    # 2. Remove space
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Remove parenthesis
    text = re.sub(r'[()]', '', text)

    return text

def vectorizer_text(corpus, reduce_dimension=False):
    """Vectorize for text data.
    Parameters
    ----------
    corpus : List[str]
        Coprus of text
    feature : str
        column name of the text feature to be vectorized
    reduce_dimension: bool
        Whether to apply PCA after Tfidf
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', lowercase=True)
    corpus = [preprocess_text(sample) for sample in corpus]
    # Preprocessing
    X_tfidf = vectorizer.fit_transform(corpus).toarray()
    if reduce_dimension:
        pca = PCA(n_components=0.9, whiten=True)                  
        X_tfidf = pca.fit_transform(X_tfidf) 
    return X_tfidf