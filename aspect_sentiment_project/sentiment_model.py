from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    if not text or len(text.strip()) == 0:
        return "NEUTRAL"
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result["label"]
    except:
        return "NEUTRAL"
