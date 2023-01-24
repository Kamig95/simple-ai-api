from typing import List

from transformers.pipelines import Pipeline
from ai.text_processors.text_processor import TextProcessor


class HuggingFaceTextProcessor(TextProcessor):
    def __init__(self, model: Pipeline):
        self._model = model

    def process(self, text: str) -> str:
        return self._model(text)[0]["label"]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [result["label"] for result in self._model(texts)]
