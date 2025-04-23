from app.scripts.run_training_evaluation_pipeline import main
from app.training.load_config import ConfigLoader
from app.training.config_update import create_temporary_config, cleanup_temp_config


def run_with_parameter_combination_over_repeats(conf_path, updated_parameters, repeats):
    sum_score = 0

    for _ in range(repeats):
        temporary_conf_path = create_temporary_config(conf_path, updated_parameters)
        temporary_conf = ConfigLoader(temporary_conf_path)
        score = main(conf=temporary_conf)
        sum_score += score
        cleanup_temp_config(temporary_conf_path)

    avg_score = sum_score / repeats

    return avg_score
