from typing import List, Any, Dict

import torch

from ai.text_processors.text_processor import TextProcessor


class PytorchTextProcessor(TextProcessor):

    def process(self, text: str) -> str:
        with torch.no_grad():
            text = torch.tensor(self._preprocessor(text))
            output = self._model(text, torch.tensor([0]))
            return self._labels[output.argmax(1).item() + 1]

    def process_batch(self, texts: List[str]) -> List[str]:
        pass
