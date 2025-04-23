from abc import abstractmethod
from PIL import Image
from torchvision import transforms  # type: ignore
from app.preprocessing.TransformationStrategy import TransformationStrategy
from app.training.config_summary import Config_Summary


class BaseImageTransform(TransformationStrategy):

    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.grayscale_transform = transforms.Grayscale(
            num_output_channels=1) if config.grayscale else transforms.Lambda(lambda x: x)

    def get_transformation_steps(self):
        transform_steps = [
            transforms.Resize((self.config.input_shape, self.config.input_shape)),
            self.grayscale_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ]

        return transform_steps

    @abstractmethod
    def process_image(self, img: Image.Image) -> Image.Image:
        pass

    def get_transform_operations(self):
        transform_steps = []
        transform_steps.append(transforms.Lambda(self.process_image))
        transform_steps += self.get_transformation_steps()

        return transforms.Compose(transform_steps)
