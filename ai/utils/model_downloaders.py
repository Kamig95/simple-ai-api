import os

from transformers.pipelines import Pipeline, pipeline

from settings import MODEL_DIR, HUGGING_FACE


def download_from_hf(model_name: str, task: str, path_to_model: str) -> Pipeline:
    classifier = pipeline(task, model=model_name)
    classifier.save_pretrained(path_to_model)

    return classifier


def load_or_download_from_hf(model_name: str, task: str) -> Pipeline:
    path_to_model = os.path.join(MODEL_DIR, HUGGING_FACE, model_name)
    if os.path.exists(path_to_model):
        return pipeline(task, model=path_to_model, tokenizer=path_to_model)
    return download_from_hf(model_name, task, path_to_model)
