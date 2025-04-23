from app.augmentation.transform_operations import BaseImageTransform
from app.training.config_summary import Config_Summary
from app.preprocessing.register_augmentation_strategies import register_transformation


@register_transformation("base")
class NoAugmentationTransform(BaseImageTransform):
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_type = config.augm_type

    def process_image(self, image):
        return image
