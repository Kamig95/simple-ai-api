from typing import Final

from ai.registry.model_registry import ModelRegistry
from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor
from ai.text_processors.text_processor import TextProcessor
from settings import HF_SENTIMENT_ANALYSIS

TEXT_ARG_NEEDED: Final[str] = "`text` argument needed"


def process_text(text: str, text_processor: TextProcessor):
    if not text:
        return TEXT_ARG_NEEDED, 400
    return text_processor.process(text)


def register_all_models() -> ModelRegistry:
    model_registry = ModelRegistry()
    model_registry.register(
        "bert_class",
        HuggingFaceTextProcessor(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            task=HF_SENTIMENT_ANALYSIS,
        ),
        "This is simple model for sentiment analysis of text",
    )
    return model_registry
