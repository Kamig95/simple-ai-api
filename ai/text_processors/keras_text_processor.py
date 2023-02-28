from typing import List

from ai.text_processors.text_processor import TextProcessor


class KerasTextProcessor(TextProcessor):
    def __init__(self, model, labels):
        self._model = model
        self._labels = labels

    def process(self, text: str) -> str:
        label_index = self._model.predict([text]).argmax(axis=-1)[0]
        return self._labels[label_index]

    def process_batch(self, texts: List[str]) -> List[str]:
        label_indices = self._model.predict(texts).argmax(axis=-1)
        return [self._labels[label_index] for label_index in label_indices]
