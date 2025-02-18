from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Uncomment if you encounter any errors due to model downloading
# import nltk
# nltk.download('vader_lexicon')

# Load Hugging Face's sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize NLTK's VADER sentiment analyzer
nltk_analyzer = SentimentIntensityAnalyzer()

# Sample sentences for sentiment analysis
sentences = [
    "I have a great news!",
    "This is the worst experience I have ever had.",
    "The movie was okay, not the best but not the worst.",
    "I am feeling really excited about my new job.",
    "The weather today is so gloomy and depressing."
]

def sentiment_huggingface(sentence):
    """Perform sentiment analysis using a transformer model."""
    return sentiment_pipeline(sentence)[0]  # Returns label and score

def sentiment_nltk(sentence):
    """Perform sentiment analysis using NLTK's VADER lexicon."""
    return nltk_analyzer.polarity_scores(sentence)  # Returns compound score

def compare_sentiments():
    """Compare results from Hugging Face and NLTK for each sentence."""
    for sentence in sentences:
        hf_result = sentiment_huggingface(sentence)
        nltk_result = sentiment_nltk(sentence)

        print("\nSentence:", sentence)
        print(f"Hugging Face Sentiment: {hf_result['label']} (Confidence: {hf_result['score']:.4f})")
        print(f"NLTK Sentiment (Compound Score): {nltk_result['compound']:.4f}")

if __name__ == "__main__":
    compare_sentiments()

