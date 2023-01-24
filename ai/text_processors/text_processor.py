from abc import ABC, abstractmethod
from typing import List


class TextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[str]:
        pass
