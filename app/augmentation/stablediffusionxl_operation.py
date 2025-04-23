from app.preprocessing.register_augmentation_strategies import register_transformation
from app.preprocessing.choose_prompts import load_prompts_in_order
from app.augmentation.transform_operations import BaseImageTransform
from app.training.config_summary import Config_Summary
import torch
from PIL import Image
import random
from diffusers import AutoPipelineForInpainting


@register_transformation("SD_XL")
class StableDiffusionXLTransform(BaseImageTransform):
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_probability = config.augm_probability
        self.pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                              torch_dtype=torch.float16, variant="fp16")
        self.generator = torch.Generator("cuda").manual_seed(92)
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_generator = load_prompts_in_order(config)
        self.mask_path = config.mask_path

    def process_image(self, img: Image.Image) -> Image.Image:
        if random.uniform(0, 1) < self.augm_probability:
            mask = Image.open(self.mask_path).convert("L")
            mask = mask.resize(img.size)
            prompt = next(self.prompt_generator)
            return self.pipe(prompt=prompt, image=img, mask_image=mask, generator=self.generator).images[0]
        return img
