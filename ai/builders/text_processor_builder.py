from abc import ABC, abstractmethod
from typing import Any, Dict

from ai.text_processors.text_processor import TextProcessor


class TextProcessorBuilder(ABC):
    def __init__(self, config: Dict):
        self._config: Dict = config
        self._model = None

    @abstractmethod
    def text_processor(self) -> TextProcessor:
        pass

    def init_model(self):
        if "model" in self._config:
            self._model = self._config["model"]

    def set_labels(self):
        if "labels" in self._config:
            self.text_processor.set_labels(self._config["labels"])

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def set_preprocessor(self):
        pass

    def create(self):
        self.init_model()
        self.load_model()
        self.set_labels()
        self.set_preprocessor()
