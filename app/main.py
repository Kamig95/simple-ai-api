from flask import Flask, request

from app.utils import process_text
from text_processors.fake_text_processor import FakeTextProcessor

app = Flask(__name__)


@app.route("/v1/input_text")
def input_text():
    text = request.args.get("text")
    return process_text(text, FakeTextProcessor())

