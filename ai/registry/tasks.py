from enum import Enum


class TaskName(str, Enum):
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    NEWS_CLASSIFICATION = "new-classification"
