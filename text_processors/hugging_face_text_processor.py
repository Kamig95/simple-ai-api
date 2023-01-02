from text_processors.text_processor import TextProcessor
from transformers import pipeline, AutoModel


class HuggingFaceTextProcessor(TextProcessor):
    def __init__(self):
        self._classifier = pipeline(
            "text-classification",
            model="./models/hugging_face/distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="./models/hugging_face/distilbert-base-uncased-finetuned-sst-2-english"
        )

    def process(self, text: str) -> str:
        return self._classifier(text)[0]['label']
