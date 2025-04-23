from app.training.load_data import load_data
from app.training.training import training_loop
from app.preprocessing.create_subset_loader import create_subset_loader
from app.models.load_model import load_model
from app.training.load_config import ConfigLoader
from app.preprocessing.get_transformation_processes import TransformationProcess
from app.evaluation.get_evaluation_metric import EvaluationMetric
import pathlib

conf_path = pathlib.Path(__file__).parent.parent / "configuration/config_training.json"
conf = ConfigLoader(config_file=conf_path)


def main(conf=conf):
    train_dir = conf.config.train_dir
    val_dir = conf.config.val_dir
    test_dir = conf.config.test_dir
    transformation_type = conf.config.augm_type

    transform_process = TransformationProcess(config_loader=conf, transformation_name=transformation_type)
    training_transformation = transform_process.get_augmentation_transform()
    validation_transformation = transform_process.get_base_transform()

    train_data_loader = load_data(train_dir, shuffle=True, transform_process=training_transformation)
    train_subset_data_loader = create_subset_loader(train_data_loader, 1)
    val_data_loader = load_data(val_dir, transform_process=validation_transformation)
    test_data_loader = load_data(test_dir, transform_process=validation_transformation)

    print(f"Data Loaders Created: length for training set: {len(train_subset_data_loader.dataset)},"
          f"length for testing set: {len(val_data_loader.dataset)}")

    grayscale = conf.config.grayscale
    model = load_model(num_classes=5, grayscale=grayscale)

    training_loop(model, train_subset_data_loader, val_data_loader, conf=conf)

    evaluation_type = conf.config.evaluation_type
    evaluation_metric = EvaluationMetric(config_loader=conf, evaluation_type=evaluation_type)
    evaluate = evaluation_metric.get_evaluation_metric()
    return evaluate(model, test_data_loader)


if __name__ == "__main__":
    main(conf)
