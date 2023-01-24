from typing import Optional, Dict, List, Any

from ai.registry.tasks import TaskName
from ai.text_processors.text_processor import TextProcessor


class ModelRegistry:
    def __init__(self):
        self._models = {}
        self._tasks: Dict[TaskName, List] = {}

    def register(
        self, model_name: str, model_object: Any, model_description: str, task: TaskName
    ):
        self._models[model_name] = {
            "model_object": model_object,
            "description": model_description,
            "task": task,
        }
        if task in self._tasks:
            self._tasks[task].append(
                {
                    "name": model_name,
                    "description": model_description,
                }
            )
        else:
            self._tasks[task] = [
                {
                    "name": model_name,
                    "description": model_description,
                }
            ]

    def get_model(self, model_name: str) -> Optional[TextProcessor]:
        if model_name in self._models:
            return self._models[model_name]["model_object"]
        return None

    def get_models(self, task: TaskName) -> List:
        return self._tasks[task]

    def get_all_models(self) -> List:
        all_models = []
        for model_name, value in self._models.items():
            all_models.append(
                {
                    "name": model_name,
                    "description": value["description"],
                    "task": value["task"],
                }
            )
        return all_models
