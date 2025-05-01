from app.preprocessing.register_augmentation_strategies import register_transformation
from app.augmentation.transform_operations import BaseImageTransform
from app.training.config_summary import Config_Summary
import torch
from PIL import Image
import random
from diffusers import StableDiffusionInstructPix2PixPipeline
from app.preprocessing.choose_prompts import load_prompts_in_order


@register_transformation("instructpix2pix")
class InstructPix2PixTransform(BaseImageTransform):
    """
    Access is gained via huggingface. Prompts are loaded in an order so each prompt is applied equally.
    Images are processed via a defined augmentation probability, applying no mask but the next prompt to the respective images.
    """
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_probability = config.augm_probability
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           torch_dtype=torch.float32)
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_generator = load_prompts_in_order(config)

    def process_image(self, img: Image.Image) -> Image.Image:
        if random.uniform(0, 1) < self.augm_probability:
            prompt = next(self.prompt_generator)
            return self.pipe(prompt=prompt, image=img).images[0]
        return img
