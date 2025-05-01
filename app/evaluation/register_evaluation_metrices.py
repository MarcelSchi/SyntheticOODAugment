import os
import importlib
from pathlib import Path

Evaluation_Registry = {}


def register_evaluation_metric(name):
    def decorator(cls):
        Evaluation_Registry[name] = cls
        return cls

    return decorator

# register all metrices that are defined in app/evaluation directory. New metrices need @register(_new_metric_) at the beginning. 
def auto_register_evaluation_metrices(module_path="app.evaluation"):
    base_path = Path(__file__).resolve().parent.parent
    module_dir = os.path.join(base_path, module_path.split(".")[-1])

    for filename in os.listdir(module_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{module_path}.{filename[:-3]}"
            importlib.import_module(module_name)
