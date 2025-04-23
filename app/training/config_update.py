import json
from pathlib import Path


def create_temporary_config(config_path, updated_parameters):
    with open(config_path, 'r') as file:
        data = json.load(file)

    data.update(updated_parameters)

    temp_config_path = Path(config_path).parent / f"temporary_{Path(config_path).stem}.json"

    with open(temp_config_path, 'w') as file:
        json.dump(data, file, indent=4)

    return temp_config_path


def cleanup_temp_config(temp_path):
    Path(temp_path).unlink(missing_ok=True)
