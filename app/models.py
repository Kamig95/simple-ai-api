import os

import keras_nlp

from ai.builders.hugging_face_text_processor_builder import HuggingFaceTextProcessorBuilder
from ai.builders.keras_text_processor_builder import KerasTextProcessorBuilder
from ai.builders.pytorch_text_processor_builder import PytorchTextProcessorBuilder
from ai.models.pytorch.text_classification_model import TextClassificationModel
from ai.registry.model_registry import ModelRegistry
from ai.registry.tasks import TaskName

from torchtext.datasets import AG_NEWS


def register_all_models() -> ModelRegistry:
    model_registry = ModelRegistry()

    huggingface_text_processor_builder = HuggingFaceTextProcessorBuilder({
        "model_filename": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": TaskName.sentiment_analysis,
    })
    huggingface_text_processor_builder.create()

    model_registry.register(
        "bert_sa_hf",
        huggingface_text_processor_builder.text_processor,
        "Simple model for sentiment analysis of text from hugging face",
        TaskName.sentiment_analysis,
    )
    keras_builder = KerasTextProcessorBuilder({
        "model": keras_nlp.models.BertClassifier.from_preset(
                "bert_tiny_en_uncased_sst2"
            ),
        "labels": {0: "NEGATIVE", 1: "POSITIVE"},
    })
    keras_builder.create()

    model_registry.register(
        "bert_sa_keras",
        keras_builder.text_processor,
        "Simple model for sentiment analysis of text from keras",
        TaskName.sentiment_analysis,
    )

    pytorch_builder = PytorchTextProcessorBuilder({
        "model": TextClassificationModel(95811, 64, 4),
        "model_filename": 'news_class_pytorch.pt',
        "labels": {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"},
        "data_iterator": AG_NEWS(split="train"),
    })
    pytorch_builder.create()
    model_registry.register(
        "news_class_pytorch",
        pytorch_builder.text_processor,
        "Simple model for text classification of AG news",
        TaskName.news_classification,
    )
    return model_registry
