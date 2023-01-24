import os
from typing import List

from transformers import pipeline
from transformers.pipelines import Pipeline
from ai.text_processors.text_processor import TextProcessor
from settings import HUGGING_FACE, MODEL_DIR


class HuggingFaceTextProcessor(TextProcessor):
    def __init__(self, model_name: str, task: str):
        self._model_name = model_name
        self._task = task
        self._path_to_model = os.path.join(MODEL_DIR, HUGGING_FACE, model_name)
        if os.path.exists(self._path_to_model):
            self._classifier = pipeline(
                task, model=self._path_to_model, tokenizer=self._path_to_model
            )
        else:
            self._classifier = self._download_model()

    def process(self, text: str) -> str:
        return self._classifier(text)[0]["label"]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [result["label"] for result in self._classifier(texts)]

    def _download_model(self) -> Pipeline:
        classifier = pipeline(self._task, model=self._model_name)
        classifier.save_pretrained(self._path_to_model)

        return classifier
