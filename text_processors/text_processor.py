from abc import ABC


class TextProcessorBase(ABC):
    def process(self, text: str) -> str:
        pass
