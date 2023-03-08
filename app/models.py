import keras_nlp
from torchtext.datasets import AG_NEWS

from ai.models.pytorch.text_classification_model import TextClassificationModel
from ai.models.pytorch.utils import get_text_pipeline
from ai.registry.model_registry import ModelRegistry
from ai.registry.tasks import TaskName
from ai.text_processors.hugging_face_text_processor import HuggingFaceTextProcessor
from ai.text_processors.keras_text_processor import KerasTextProcessor
from ai.text_processors.pytorch_text_processor import PytorchTextProcessor
from ai.utils.model_downloaders import load_or_download_from_hf


def register_all_models() -> ModelRegistry:
    model_registry = ModelRegistry()

    model_registry.register(
        "bert_sa_hf",
        HuggingFaceTextProcessor(
            load_or_download_from_hf(
                model_filename="distilbert-base-uncased-finetuned-sst-2-english",
                task=TaskName.SENTIMENT_ANALYSIS,
            ),
            result_key="label",
        ),
        "Model for sentiment analysis of text from hugging face",
        TaskName.SENTIMENT_ANALYSIS,
    )

    model_registry.register(
        "bert_qa_hf",
        HuggingFaceTextProcessor(
            load_or_download_from_hf(model_filename="gpt2", task="text-generation"),
            result_key="generated_text",
        ),
        "Model for text generation from hugging face",
        TaskName.TEXT_GENERATION,
    )

    model_registry.register(
        "translation_en_to_fr_hf",
        HuggingFaceTextProcessor(
            load_or_download_from_hf(model_filename=None, task="translation_en_to_fr"),
            result_key="translation_text",
        ),
        "Model for translation from en to fr from hugging face",
        TaskName.TRANSLATION,
    )

    model_registry.register(
        "bert_sa_keras",
        KerasTextProcessor(
            model=keras_nlp.models.BertClassifier.from_preset(
                "bert_tiny_en_uncased_sst2"
            ),
            labels={0: "NEGATIVE", 1: "POSITIVE"},
        ),
        "Model for sentiment analysis of text from keras",
        TaskName.SENTIMENT_ANALYSIS,
    )

    model_registry.register(
        "news_class_pytorch",
        PytorchTextProcessor(
            model=TextClassificationModel(95811, 64, 4),
            model_filename="news_class_pytorch.pt",
            labels={1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"},
            preprocessor=get_text_pipeline(AG_NEWS(split="train")),  # pylint: disable=E1120
        ),
        "Model for text classification of AG news",
        TaskName.NEWS_CLASSIFICATION,
    )
    return model_registry
