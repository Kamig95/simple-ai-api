from typing import List, Any

from ai.text_processors.text_processor import TextProcessor


class KerasTextProcessor(TextProcessor):

    def process(self, text: str) -> str:
        label_index = self._model.predict([text]).argmax(axis=-1)[0]
        return self._labels[label_index]

    def process_batch(self, texts: List[str]) -> List[str]:
        label_indices = self._model.predict(texts).argmax(axis=-1)
        return [self._labels[label_index] for label_index in label_indices]
