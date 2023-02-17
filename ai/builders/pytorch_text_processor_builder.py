import os
from typing import Dict

import torch
from torchtext.vocab import build_vocab_from_iterator

from ai.builders.text_processor_builder import TextProcessorBuilder
from ai.models.pytorch.utils import yield_tokens, tokenizer
from ai.text_processors.pytorch_text_processor import PytorchTextProcessor
from ai.text_processors.text_processor import TextProcessor
from settings import MODEL_DIR, PYTORCH


class PytorchTextProcessorBuilder(TextProcessorBuilder):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._text_processor = PytorchTextProcessor()

    @property
    def text_processor(self) -> TextProcessor:
        return self._text_processor

    def load_model(self):
        path_to_model = os.path.join(MODEL_DIR, PYTORCH, self._config["model_filename"])
        self._model.load_state_dict(torch.load(path_to_model))
        self._model.eval()
        self._text_processor.set_model(self._model)

    def set_preprocessor(self):
        vocab = build_vocab_from_iterator(yield_tokens(self._config['data_iterator']), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        self._text_processor.set_preprocessor(lambda x: vocab(tokenizer(x)))
