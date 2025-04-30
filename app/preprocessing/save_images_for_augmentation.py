import os
from tqdm import tqdm
from torchvision.utils import save_image  # type: ignore
from torch.utils.data import DataLoader

# save all images that are augmented during the augmentation process. Use an augmented dataset for later training
def save_augmented_images(dataset, augmentation_type='base', output_dir='augmented_images'):
    output_dir = os.path.join(output_dir, augmentation_type)
    os.makedirs(output_dir, exist_ok=True)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Saving augmented images for augmentation type: '{augmentation_type}'")
    for idx, (image, label) in enumerate(tqdm(data_loader, desc="Saving Images")):
        class_name = dataset.classes[label.item()]

        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_save_path = os.path.join(class_dir, f"image_{idx}.png")

        save_image(image[0], image_save_path)

    print(f"Augmented images saved in '{output_dir}'")
