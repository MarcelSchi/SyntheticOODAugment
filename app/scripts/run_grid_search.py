import pathlib
from app.optimization.save_results import save_results_for_grid_search
from app.optimization.single_run_with_parameter_combination import run_with_parameter_combination_over_repeats
from app.optimization.generate_parameter_combinations import generate_parameter_combinations


def run_experiment_with_grid(conf_path, param_grid, repeats=5, output_file="results/experiment_results.json"):
    results = {}
    parameter_combinations = generate_parameter_combinations(param_grid)

    for updated_parameters in parameter_combinations:
        print(f"Running experiments with parameters: {updated_parameters}")

        avg_score = run_with_parameter_combination_over_repeats(conf_path, updated_parameters, repeats)
        results[str(updated_parameters)] = avg_score

        print(f"Average accuracy for parameters {updated_parameters}: {avg_score:.2f} \n")

    save_results_for_grid_search(results, output_file)

    return results


if __name__ == "__main__":
    conf_path = pathlib.Path(__file__).parent.parent / "configuration/config_training.json"
    param_grid = {
        "augm_probability": [0],
        "number_epochs": [2],
        "learning_rate": [0.001],
        "batch_size": [32, 64],
    }

    output_file = pathlib.Path(__file__).parent.parent.parent / "results/experiment_results.json"

    run_experiment_with_grid(conf_path, param_grid, repeats=2, output_file=str(output_file))
