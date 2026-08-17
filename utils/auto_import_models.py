import os
import importlib

def import_models() -> dict:

    models_dict = {}

    models_path = "models"

    for file in os.listdir(models_path):

        if file in ("__init__.py", "basic_algorithms_class.py"):
            continue

        if not file.endswith("_model.py"):
            continue

        module_name = file[:-3]

        module = importlib.import_module(f"models.{module_name}")

        cls = getattr(module, module_name)

        model = cls()

        models_dict[model.model_name] = model

    sorted_models_dict =  dict(sorted(models_dict.items()))

    return sorted_models_dict
