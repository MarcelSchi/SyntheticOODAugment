import requests
import os
from PIL import Image


def extract_image_from_api_online(output):
    output_url = output.url

    response = requests.get(output_url)
    response.raise_for_status()

    output_dir = "augmented_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "augmented_image.png")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return Image.open(output_path)
