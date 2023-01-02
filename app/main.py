from flask import Flask, request

from app.utils import process_text
from text_processors.fake_text_processor import FakeTextProcessor
from text_processors.hugging_face_text_processor import HuggingFaceTextProcessor

app = Flask(__name__)
hugging_face_text_processor = HuggingFaceTextProcessor()


@app.route("/v1/input_text")
def input_text():
    text = request.args.get("text")
    return process_text(text, FakeTextProcessor())

@app.route("/v1/hugging_face")
def hugging_face():
    text = request.args.get("text")
    return process_text(text, hugging_face_text_processor)
