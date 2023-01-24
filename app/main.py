from fastapi import FastAPI

from ai.registry.tasks import TaskName
from app.models import register_all_models
from app.utils import process_text
from ai.text_processors.fake_text_processor import FakeTextProcessor


app = FastAPI()

model_registry = register_all_models()


@app.get("/input_text")
async def input_text(text: str):
    return {"output": process_text(text, FakeTextProcessor())}


@app.get("/predict/{model}")
async def predict(model: str, text: str):
    return {"prediction": process_text(text, model_registry.get_model(model))}


@app.get("/models")
async def all_models():
    return {"models": model_registry.get_all_models()}


@app.get("/models/{task}")
async def models(task: TaskName):
    return {"models": model_registry.get_models(TaskName[task])}
