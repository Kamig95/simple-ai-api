from flask import Flask, request

from app.models import register_all_models
from app.utils import process_text
from ai.text_processors.fake_text_processor import FakeTextProcessor

app = Flask(__name__)

model_registry = register_all_models()


@app.route("/v1/input_text")
def input_text():
    text = request.args.get("text")
    return process_text(text, FakeTextProcessor())


@app.route("/v1/sentiment_analysis")
def hugging_face():
    text = request.args.get("text")
    model = request.args.get("model")
    return process_text(text, model_registry.get_model(model))
