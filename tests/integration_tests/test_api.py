from fastapi.testclient import TestClient

from app.main import app
from tests.integration_tests.data import MODELS_RESPONSE, MODELS_TRANSLATION_RESPONSE

client = TestClient(app)


def test_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == MODELS_RESPONSE


def test_models_task():
    task = "translation"
    response = client.get(f"/models/{task}")
    assert response.status_code == 200
    assert response.json() == MODELS_TRANSLATION_RESPONSE


def test_predict_hugging_face():
    model = "bert_sa_hf"
    response = client.get(f"/predict/{model}", params={"text": "This is great movie."})
    assert response.status_code == 200
    assert response.json() == {"prediction": "POSITIVE"}
