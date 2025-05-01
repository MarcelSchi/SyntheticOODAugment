from app.preprocessing.register_augmentation_strategies import register_transformation
from app.augmentation.transform_operations import BaseImageTransform
from app.preprocessing.choose_prompts import load_prompts_in_order
from app.training.config_summary import Config_Summary
from app.augmentation.replicate_helper_function import extract_image_from_api_online
import replicate
import os
from PIL import Image
import random
import io

@register_transformation("Flux_Fill_Pro")
class FluxFillProTransform(BaseImageTransform):
    def __init__(self, config: Config_Summary):
        super().__init__(config)
        self.augm_probability = config.augm_probability
        self.mask_path = config.mask_path
        self.prompt_generator = load_prompts_in_order(config)
        # Flux needs an API access -> url is provided. 
        self.api_url = "black-forest-labs/flux-fill-pro"

    def process_image(self, img: Image.Image) -> Image.Image:
        if random.uniform(0, 1) < self.augm_probability:
            prompt = next(self.prompt_generator)

            # API key has to be set as an environmental variable.
            api_key = os.getenv("REPLICATE_API_TOKEN")
            if not api_key:
                raise ValueError("API Key not found! Set REPLICATE_API_TOKEN in your environment variables.")

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            with open(self.mask_path, "rb") as mask_file:
                output = replicate.run(
                    self.api_url,
                    input={
                        "mask": mask_file,
                        "image": img_bytes,
                        "prompt": prompt
                    }
                )

            # replicate helper function to gain image in appropriate format
            if isinstance(output, replicate.helpers.FileOutput):
                extracted_img = extract_image_from_api_online(output)
                return extracted_img

            else:
                raise TypeError(f"Unexpected output type from Replicate: {type(output)}")

        return img
