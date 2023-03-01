import os
from typing import List

import torch
from torchtext.vocab import build_vocab_from_iterator

from ai.models.pytorch.utils import yield_tokens, tokenizer
from ai.text_processors.text_processor import TextProcessor
from settings import MODEL_DIR, PYTORCH


class PytorchTextProcessor(TextProcessor):
    def __init__(self, model, model_filename, labels, data_iterator):
        self._model = model
        self._model_filename = model_filename
        self._labels = labels
        self._data_iterator = data_iterator
        self._preprocessor = self._create_preprocessor()

    def process(self, text: str) -> str:
        with torch.no_grad():
            text = torch.tensor(self._preprocessor(text))
            output = self._model(text, torch.tensor([0]))
            return self._labels[output.argmax(1).item() + 1]

    def process_batch(self, texts: List[str]) -> List[str]:
        pass

    def load_model(self):
        path_to_model = os.path.join(MODEL_DIR, PYTORCH, self._model_filename)
        self._model.load_state_dict(torch.load(path_to_model))
        self._model.eval()

    def _create_preprocessor(self):
        vocab = build_vocab_from_iterator(
            yield_tokens(self._data_iterator), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        return lambda x: vocab(tokenizer(x))
