from flask import Flask, request

from app.utils import process_text, register_all_models
from ai.text_processors.fake_text_processor import FakeTextProcessor

app = Flask(__name__)

model_registry = register_all_models()


@app.route("/v1/input_text")
def input_text():
    text = request.args.get("text")
    return process_text(text, FakeTextProcessor())


@app.route("/v1/hugging_face")
def hugging_face():
    text = request.args.get("text")
    model = request.args.get("model")
    return process_text(text, model_registry.get_model(model))
