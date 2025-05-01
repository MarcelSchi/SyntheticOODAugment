import os
import importlib
from pathlib import Path
from app.preprocessing.TransformationStrategy import TransformationStrategy

Transformation_Registry = {}


def register_transformation(name):

    def decorator(cls):
        Transformation_Registry[name] = cls
        return cls

    return decorator


# goes through all python scripts to register all available augmentation strategies.
# New strategies can be added dynamically by adding a registration with a keyword -> choose augmentation strategy via configuration file
def auto_register_transformations(module_path="app.augmentation"):
    base_path = Path(__file__).resolve().parent.parent
    module_dir = os.path.join(base_path, module_path.split(".")[-1])

    for filename in os.listdir(module_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{module_path}.{filename[:-3]}"
            importlib.import_module(module_name)

    for cls in TransformationStrategy.__subclasses__():
        if cls not in Transformation_Registry.values():
            name = getattr(cls, "name", cls.__name__.lower())
            Transformation_Registry[name] = cls
