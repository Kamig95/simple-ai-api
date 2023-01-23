from typing import Final

from ai.text_processors.text_processor import TextProcessor

TEXT_ARG_NEEDED: Final[str] = "`text` argument needed"


def process_text(text: str, text_processor: TextProcessor):
    if not text:
        return TEXT_ARG_NEEDED, 400
    return text_processor.process(text)
