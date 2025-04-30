import pathlib
from app.training.load_config import ConfigLoader
from app.models.load_trained_model import load_trained_model
from app.preprocessing.get_transformation_processes import TransformationProcess
from app.training.load_data import load_data
from app.evaluation.get_evaluation_metric import EvaluationMetric
from app.training.config_update import create_temporary_config, cleanup_temp_config

conf_path = pathlib.Path(__file__).parent.parent / "configuration/config_training.json"
conf = ConfigLoader(config_file=conf_path)


def evaluate_model_on_test_set(model_path, new_test_dir=None, conf=conf):

    temp_config_path = None
    if new_test_dir is not None:

        updated_parameters = {"test_dir": new_test_dir}
        temp_config_path = create_temporary_config(conf_path, updated_parameters)
        conf = ConfigLoader(config_file=temp_config_path)

    model = load_trained_model(model_path)

    transformation_type = conf.config.augm_type
    transform_process = TransformationProcess(config_loader=conf, transformation_name=transformation_type)
    test_transformation = transform_process.get_base_transform()

    test_dir = conf.config.test_dir
    new_test_data_loader = load_data(test_dir, transform_process=test_transformation)

    evaluation_type = conf.config.evaluation_type
    evaluation_metric = EvaluationMetric(config_loader=conf, evaluation_type=evaluation_type)
    evaluate = evaluation_metric.get_evaluation_metric()

    score = evaluate(model, new_test_data_loader)
    print(f"Test set: {conf.config.test_dir} with model: {model_path}, score = {score}")

    if temp_config_path is not None:
        cleanup_temp_config(temp_config_path)

    return score


if __name__ == "__main__":
    test_dataset_dir = "app/data/test_dataset/"
    evaluate_model_on_test_set(
        "efficientnet.pth",
        new_test_dir=test_dataset_dir
    )
