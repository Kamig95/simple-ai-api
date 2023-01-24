import keras_nlp

from ai.registry.model_registry import ModelRegistry
from ai.registry.tasks import TaskName
from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor
from ai.text_processors.keras_text_processor import KerasTextProcessor
from ai.utils.model_downloaders import load_or_download_from_hf
from settings import HF_SENTIMENT_ANALYSIS


def register_all_models() -> ModelRegistry:
    model_registry = ModelRegistry()
    model_registry.register(
        "bert_sa_hf",
        HuggingFaceTextProcessor(
            model=load_or_download_from_hf(
                model_name="distilbert-base-uncased-finetuned-sst-2-english",
                task=HF_SENTIMENT_ANALYSIS,
            )
        ),
        "This is simple model for sentiment analysis of text from hugging face",
        TaskName.sentiment_analysis,
    )
    model_registry.register(
        "bert_sa_keras",
        KerasTextProcessor(
            model=keras_nlp.models.BertClassifier.from_preset(
                "bert_tiny_en_uncased_sst2"
            ),
            labels=["NEGATIVE", "POSITIVE"],
        ),
        "This is simple model for sentiment analysis of text from keras",
        TaskName.sentiment_analysis,
    )
    return model_registry
