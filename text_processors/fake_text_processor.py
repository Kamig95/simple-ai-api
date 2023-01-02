from text_processors.text_processor import TextProcessor


class FakeTextProcessor(TextProcessor):
    def process(self, text: str) -> str:
        return text
