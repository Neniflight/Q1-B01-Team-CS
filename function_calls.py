from textblob import TextBlob
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


def emotionality_analyzer(text:str):
    """Analyzes text based on its emotionality and exaggeration"""
    emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
    sia = SentimentIntensityAnalyzer()

    emotions = emotion_analyzer(text)
    emotion_scores = {e['label']: e['score'] for e in emotions[0]}\
    # chooses the max emotion from joy, anger, sadness, or surprise
    max_emotion_score = max(emotion_scores.values())

    sentiment = sia.polarity_scores(text)
    sentiment_intensity = abs(sentiment['compound'])

    exaggeration_score = 0
    words = text.split()
    superlatives = ["best", "worst", "most", "least", "unbelievably", "extremely", "incredibly", "very", "absolutely"]
    for word in words:
        if word.lower() in superlatives:
            exaggeration_score += 1

    exaggeration_ratio = exaggeration_score / len(words)
    exaggeration_ratio = min(exaggeration_ratio * 10, 1)

    final_score = ((1/3) * max_emotion_score) + ((1/3) * sentiment_intensity) + ((1/3) * exaggeration_ratio)
    scaled_score = round(final_score * 10, 2)
    print(f"emotionalaity_score: {scaled_score}")
    return scaled_score

# if __name__ == "__main__":
#    print(emotionality_analyzer("Shocking Discovery: Scientists Unveil a Secret That Will Change Humanity Forever!"))