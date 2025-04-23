from torchvision import datasets  # type: ignore
from torch.utils.data import DataLoader
from app.preprocessing.get_transformation_processes import TransformationProcess
import pathlib
from app.training.load_config import ConfigLoader

config_path = pathlib.Path(__file__).parent.parent / "configuration/config_training.json"
conf = ConfigLoader(config_file=config_path)
augmentation_type = conf.config.augm_type

transform_process = TransformationProcess(config_loader=conf, transformation_name=augmentation_type)
base_transformation = transform_process.get_base_transform()


def load_data(data_directory, shuffle=False, transform_process=base_transformation):
    _dataset = datasets.ImageFolder(root=data_directory, transform=transform_process)
    batch_size = conf.config.batch_size

    data_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    return data_loader
