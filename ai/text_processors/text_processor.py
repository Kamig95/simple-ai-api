from abc import ABC, abstractmethod
from typing import List


class TextProcessor(ABC):
    def __init__(self):
        self._model = None
        self._labels = None
        self._preprocessor = None

    def set_model(self, model):
        self._model = model

    def set_labels(self, labels):
        self._labels = labels

    def set_preprocessor(self, preprocessor):
        self._preprocessor = preprocessor

    @abstractmethod
    def process(self, text: str) -> str:
        pass

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[str]:
        pass
