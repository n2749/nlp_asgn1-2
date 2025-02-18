import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Uncomment if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

from get_paragraph import get_paragraph_from_war_and_peace


### Tokenization
def tokenize_nltk(text: str):
    """Tokenize text using NLTK."""
    return word_tokenize(text)


def tokenize_spacy(text: str):
    """Tokenize text using spaCy."""
    return [token.text for token in nlp(text)]


### Lemmatization
def lemmatize_nltk(nltk_tokens):
    """Lemmatize tokens using NLTK."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in nltk_tokens]


def lemmatize_spacy(text: str):
    """Lemmatize tokens using spaCy."""
    return [token.lemma_ for token in nlp(text)]


### Stopword Removal
def stopwords_nltk(nltk_tokens):
    """Remove stopwords using NLTK."""
    stop_words = set(stopwords.words('english'))
    return [word for word in nltk_tokens if word.lower() not in stop_words]


def stopwords_spacy(text: str):
    """Remove stopwords using spaCy."""
    return [token.text for token in nlp(text) if not token.is_stop]


### Comparison Function
def compare_results(nltk_result, spacy_result, label):
    """Compare results between NLTK and spaCy."""
    print(f"\nComparison for {label}:")
    print(f"NLTK: {nltk_result[:10]} ... (Total: {len(nltk_result)})")
    print(f"spaCy: {spacy_result[:10]} ... (Total: {len(spacy_result)})")


### Main Execution
def main():
    text = get_paragraph_from_war_and_peace()
    print(f"\nSample Paragraph:\n\"\"\"\n{text}\n\"\"\"\n")

    # Tokenization
    nltk_tokens = tokenize_nltk(text)
    spacy_tokens = tokenize_spacy(text)
    compare_results(nltk_tokens, spacy_tokens, "Tokenization")

    # Lemmatization
    nltk_lemmas = lemmatize_nltk(nltk_tokens)
    spacy_lemmas = lemmatize_spacy(text)
    compare_results(nltk_lemmas, spacy_lemmas, "Lemmatization")

    # Stopword Removal
    nltk_filtered = stopwords_nltk(nltk_tokens)
    spacy_filtered = stopwords_spacy(text)
    compare_results(nltk_filtered, spacy_filtered, "Stopword Removal")


if __name__ == "__main__":
    main()

