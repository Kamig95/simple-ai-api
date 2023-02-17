from enum import Enum


class TaskName(str, Enum):
    sentiment_analysis = "sentiment-analysis"
    news_classification = "new-classification"
