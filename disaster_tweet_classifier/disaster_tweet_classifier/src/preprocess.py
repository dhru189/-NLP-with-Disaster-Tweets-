"""
Disaster Tweet Classifier - Preprocessing Module
Implements the required classical pipeline steps:
1. Clean tweets using regex (remove URLs, hashtags, mentions)
2. Tokenize and preprocess using NLTK
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean tweet text using regex:
    - Remove URLs
    - Remove mentions (@user)
    - Remove hashtags symbols (keep the word)
    - Remove HTML entities
    - Remove punctuation / special chars
    - Lowercase
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&\w+;', ' ', text)
    
    # Remove numbers (optional, keep for now)
    # text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> list:
    """
    Tokenize using NLTK and optionally remove stopwords + lemmatize.
    """
    if not text:
        return []
    
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def preprocess_pipeline(text: str) -> str:
    """
    Full pipeline: clean → tokenize → stopword removal → lemmatize → join back to string
    (ready for TF-IDF)
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_preprocess(cleaned)
    return " ".join(tokens)


def get_preprocessing_steps(text: str) -> dict:
    """
    Return step-by-step intermediate results for the playground UI.
    """
    original = text
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    no_stop = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatized = [lemmatizer.lemmatize(t) for t in no_stop]
    
    return {
        "original": original,
        "cleaned": cleaned,
        "tokens": tokens,
        "no_stopwords": no_stop,
        "lemmatized": lemmatized,
        "final": " ".join(lemmatized)
    }
