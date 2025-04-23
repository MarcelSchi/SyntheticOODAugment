import json


def save_results_for_grid_search(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
