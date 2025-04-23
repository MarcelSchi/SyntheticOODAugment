import random
from torchvision.transforms import AutoAugment, AutoAugmentPolicy  # type: ignore
from app.augmentation.transform_operations import BaseImageTransform
from app.training.config_summary import Config_Summary
from app.preprocessing.register_augmentation_strategies import register_transformation


@register_transformation("autoaugment")
class AugmentTransform(BaseImageTransform):
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_probability = config.augm_probability
        self.apply_auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

    def process_image(self, image):
        if random.uniform(0, 1) < self.config.augm_probability:
            return self.apply_auto_augment(image)
        return image
