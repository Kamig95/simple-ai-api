import os
from typing import List

import torch

from ai.text_processors.text_processor import TextProcessor
from settings import MODEL_DIR, PYTORCH


class PytorchTextProcessor(TextProcessor):
    def __init__(self, model, model_filename, labels, preprocessor):
        self._model = model
        self._model_filename = model_filename
        self._labels = labels
        self._preprocessor = preprocessor

    def process(self, text: str) -> str:
        with torch.no_grad():
            text = torch.tensor(self._preprocessor(text))
            output = self._model(text, torch.tensor([0]))
            return self._labels[output.argmax(1).item() + 1]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(text) for text in texts]

    def load_model(self):
        path_to_model = os.path.join(MODEL_DIR, PYTORCH, self._model_filename)
        self._model.load_state_dict(torch.load(path_to_model))
        self._model.eval()
