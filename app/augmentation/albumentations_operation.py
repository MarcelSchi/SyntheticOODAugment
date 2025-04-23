import albumentations as alb
import random
import numpy as np
from PIL import Image
from app.augmentation.transform_operations import BaseImageTransform
from app.training.config_summary import Config_Summary
from app.preprocessing.register_augmentation_strategies import register_transformation


@register_transformation("albumentations")
class AlbumentationsTransform(BaseImageTransform):
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_probability = config.augm_probability
        self.transform = alb.Compose([
            alb.HorizontalFlip(p=0.1),
            alb.RandomBrightnessContrast(p=0.5),
            alb.Rotate(limit=20, p=0.3),
            alb.GaussianBlur(blur_limit=(3, 7), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=30, p=0.2),
            alb.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3)
        ])

    def process_image(self, image: Image.Image) -> Image.Image:
        if random.uniform(0, 1) < self.augm_probability:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_np_aug = augmented["image"]
            return Image.fromarray(image_np_aug)

        return image
