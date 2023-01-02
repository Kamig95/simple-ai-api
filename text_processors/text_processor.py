from abc import ABC, abstractmethod


class TextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass
