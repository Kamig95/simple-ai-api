import os
from typing import List

from transformers.pipelines import Pipeline, pipeline
from ai.text_processors.text_processor import TextProcessor
from settings import HUGGING_FACE, MODEL_DIR


class HuggingFaceTextProcessor(TextProcessor):
    def __init__(self, model_filename: str, task: str, result_key: str):
        self._model_filename = model_filename
        self._task = task
        self._result_key = result_key
        self._model = self.load_model()

    def process(self, text: str) -> str:
        return self._model(text)[0][self._result_key]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [result[self._result_key] for result in self._model(texts)]

    def load_model(self):
        if self._model_filename:
            model_name = self._model_filename
        else:
            model_name = self._task
        path_to_model = os.path.join(MODEL_DIR, HUGGING_FACE, model_name)
        if os.path.exists(path_to_model):
            model = pipeline(self._task, model=path_to_model, tokenizer=path_to_model)
        else:
            model = self._download_from_hf(path_to_model)
        return model

    def _download_from_hf(self, path_to_model: str) -> Pipeline:
        if self._model_filename is None:
            model = pipeline(self._task)
        else:
            model = pipeline(model=self._model_filename)
        model.save_pretrained(path_to_model)

        return model
