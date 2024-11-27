from textblob import TextBlob
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def final_factuality_factor_score(microfactor_1:float, microfactor_2:float, microfactor_3:float):
    """Averages the microfactors from a single factuality factor. This function should be used when combining into an overall score.
    
    Args:
        microfactor_1: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
        microfactor_2: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
        microfactor_3: A float value from 1 to 10 that represents the score of the microfactor of a factuality factor. 
    """
    score = (microfactor_1 + microfactor_2 + microfactor_3) / 3
    print(f"factuality_score: {score}")
    return score

def emotion_analyzer(text:str):
    """Analyzes text based on its emotionality and exaggeration. This function should be used for the microfactor Emotion Analysis and the output is already scaled from 1 to 10.
    
    Args:
        text: A string value that represents the article we are grading on. 
    """
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

    final_score = ((0.4) * max_emotion_score) + ((0.4) * sentiment_intensity) + ((0.2) * exaggeration_ratio)
    scaled_score = round(final_score * 10, 2)
    print(f"emotion_score: {scaled_score}")
    return scaled_score

# if __name__ == "__main__":
#    print(emotionality_analyzer("Shocking Discovery: Scientists Unveil a Secret That Will Change Humanity Forever!"))