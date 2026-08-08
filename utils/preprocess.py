"""
Text Preprocessing Module for Customer Support Ticket System.
Includes lowercasing, URL/punctuation removal, tokenization, stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
_NLTK_RESOURCES = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
for resource in _NLTK_RESOURCES:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# Initialize Lemmatizer and Stopwords set
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
        "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
        "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
        "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
        "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
        "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
        "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
        "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
        "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans input text by lowercasing, removing URLs, HTML tags, special chars, punctuation, and extra whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters and digits
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    Cleaning -> Tokenization -> Stopword Removal -> Lemmatization -> Joined String.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    # Filter stopwords and short tokens, apply lemmatization
    processed_tokens = []
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 1:
            try:
                lemma = lemmatizer.lemmatize(word)
            except Exception:
                lemma = word
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)
