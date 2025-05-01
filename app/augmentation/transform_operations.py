from abc import abstractmethod
from PIL import Image
from torchvision import transforms  # type: ignore
from app.preprocessing.TransformationStrategy import TransformationStrategy
from app.training.config_summary import Config_Summary


class BaseImageTransform(TransformationStrategy):
    """
    This function depicts a basic transformation which applies all necessary preprocessing steps.
    Every augmentation operation function can inherit from the BaseImageTransform() function.
    This saves time and allows a modular, dynamic usage of a variety of augmentation strategies.
    """

    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.grayscale_transform = transforms.Grayscale(
            num_output_channels=1) if config.grayscale else transforms.Lambda(lambda x: x)

    # transformation steps are necessary for an appropriate image input
    def get_transformation_steps(self):
        transform_steps = [
            transforms.Resize((self.config.input_shape, self.config.input_shape)),
            self.grayscale_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ]

        return transform_steps

    # abstract method can be used for inheritage by other functions. The base transformation 
    # does not need any additional processing
    @abstractmethod
    def process_image(self, img: Image.Image) -> Image.Image:
        pass

    # this function is used by all augmentation operations to chosse the depicted function in the config file
    def get_transform_operations(self):
        transform_steps = []
        transform_steps.append(transforms.Lambda(self.process_image))
        transform_steps += self.get_transformation_steps()

        return transforms.Compose(transform_steps)
