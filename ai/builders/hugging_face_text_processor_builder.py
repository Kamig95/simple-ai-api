import os
from typing import Dict

from transformers import pipeline, Pipeline

from ai.builders.text_processor_builder import TextProcessorBuilder
from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor
from ai.text_processors.text_processor import TextProcessor
from settings import MODEL_DIR, HUGGING_FACE


class HuggingFaceTextProcessorBuilder(TextProcessorBuilder):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._text_processor = HuggingFaceTextProcessor()

    @property
    def text_processor(self) -> TextProcessor:
        return self._text_processor

    def load_model(self):
        path_to_model = os.path.join(
            MODEL_DIR, HUGGING_FACE, self._config["model_filename"]
        )
        if os.path.exists(path_to_model):
            model = pipeline(
                self._config["task"], model=path_to_model, tokenizer=path_to_model
            )
        else:
            model = self._download_from_hf(path_to_model)
        self._text_processor.set_model(model)

    def set_preprocessor(self):
        pass

    def _download_from_hf(self, path_to_model: str) -> Pipeline:
        classifier = pipeline(
            self._config["task"], model=self._config["model_filename"]
        )
        classifier.save_pretrained(path_to_model)

        return classifier
