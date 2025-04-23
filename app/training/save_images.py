import os
from torchvision.utils import save_image  # type: ignore


def save_images_from_epoch(epoch, inputs, augmentation_type='base', output_dir='saved_images'):
    base_output_dir = os.path.join(output_dir, augmentation_type)
    os.makedirs(base_output_dir, exist_ok=True)

    epoch_dir = os.path.join(base_output_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    index_for_images = len(os.listdir(epoch_dir))

    for i, img in enumerate(inputs):
        img_name = f'image_{index_for_images}.png'
        save_path = os.path.join(epoch_dir, img_name)
        save_image(img.cpu(), save_path)
        index_for_images += 1
