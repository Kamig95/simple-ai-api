from typing import List

from ai.text_processors.text_processor import TextProcessor


class HuggingFaceTextProcessor(TextProcessor):
    def __init__(self, model, result_key: str):
        self._model = model
        self._result_key = result_key

    def process(self, text: str) -> str:
        return self._model(text)[0][self._result_key]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [result[self._result_key] for result in self._model(texts)]
