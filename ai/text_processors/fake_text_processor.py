from typing import List

from ai.text_processors.text_processor import TextProcessor


class FakeTextProcessor(TextProcessor):
    def process_batch(self, texts: List[str]) -> List[str]:
        return texts

    def process(self, text: str) -> str:
        return text
