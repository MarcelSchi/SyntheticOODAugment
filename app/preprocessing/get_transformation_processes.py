from app.training.load_config import ConfigLoader
from app.preprocessing.register_augmentation_strategies import Transformation_Registry, auto_register_transformations

# function to search for all augmentation strategies that are defined in the app.augmentation directory
# helps to dynamically add new strategies by registering them in a new registry
class TransformationProcess:
    def __init__(self, config_loader: ConfigLoader, transformation_name):
        auto_register_transformations(module_path="app.augmentation")

        self.config = config_loader.get_config()
        # ensures that a augmentation strategy is defined in the configuration file actually exists.  
        if transformation_name not in Transformation_Registry:
            available_transformations = ", ".join(Transformation_Registry.keys())
            # If depicted string does not exist: Provide user a list of all existing augmentation strategies.
            raise ValueError(
                f"Transformation '{transformation_name}' not registered. "
                f"Please check all available transformations: {available_transformations}."
            )
        transformation_class = Transformation_Registry[transformation_name]
        self.transformation = transformation_class(self.config)

    def get_augmentation_transform(self):
        return self.transformation.get_transform_operations()

    def get_base_transform(self):
        base_transform_class = Transformation_Registry["base"]
        base_transform = base_transform_class(self.config)
        return base_transform.get_transform_operations()
