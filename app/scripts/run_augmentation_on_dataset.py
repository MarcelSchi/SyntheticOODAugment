from torchvision import datasets  # type: ignore
import pathlib
from app.training.load_config import ConfigLoader
from app.preprocessing.get_transformation_processes import TransformationProcess
from app.preprocessing.save_images_for_augmentation import save_augmented_images

conf_path = pathlib.Path(__file__).parent.parent / "configuration/config_training.json"
conf = ConfigLoader(config_file=conf_path)


def run_augmentation_on_dataset(conf=conf):
    augmentation_type = conf.config.augm_type
    dataset_directory = conf.config.test_dir

    transformation = TransformationProcess(config_loader=conf, transformation_name=augmentation_type)
    augmentation_process = transformation.get_augmentation_transform()

    _dataset = datasets.ImageFolder(root=dataset_directory, transform=augmentation_process)

    save_augmented_images(dataset=_dataset, augmentation_type=augmentation_type)


if __name__ == "__main__":
    run_augmentation_on_dataset(conf)
