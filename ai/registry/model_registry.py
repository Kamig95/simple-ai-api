from typing import Optional

from ai.text_processors.text_processor import TextProcessor


class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, model_name, model_object, model_description):
        self._models[model_name] = {
            "model_object": model_object,
            "description": model_description,
        }

    def get_model(self, model_name: str) -> Optional[TextProcessor]:
        if model_name in self._models:
            return self._models[model_name]["model_object"]
        return None
