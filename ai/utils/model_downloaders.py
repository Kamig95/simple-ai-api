import os

from transformers.pipelines import Pipeline, pipeline

from settings import MODEL_DIR, HUGGING_FACE


def download_from_hf(model_filename: str, task: str, path_to_model: str) -> Pipeline:
    if model_filename is None:
        model = pipeline(task)
    else:
        model = pipeline(model=model_filename)
    model.save_pretrained(path_to_model)

    return model


def load_or_download_from_hf(model_filename: str, task: str) -> Pipeline:
    if model_filename:
        model_name = model_filename
    else:
        model_name = task
    path_to_model = os.path.join(MODEL_DIR, HUGGING_FACE, model_name)
    if os.path.exists(path_to_model):
        model = pipeline(task, model=path_to_model, tokenizer=path_to_model)
    else:
        model = download_from_hf(model_filename, task, path_to_model)
    return model
