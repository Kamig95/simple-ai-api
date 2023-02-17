from typing import Dict

from ai.builders.text_processor_builder import TextProcessorBuilder
from ai.text_processors.keras_text_processor import KerasTextProcessor
from ai.text_processors.text_processor import TextProcessor


class KerasTextProcessorBuilder(TextProcessorBuilder):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._text_processor = KerasTextProcessor()

    @property
    def text_processor(self) -> TextProcessor:
        return self._text_processor

    def load_model(self):
        self._text_processor.set_model(self._model)

    def set_preprocessor(self):
        pass
