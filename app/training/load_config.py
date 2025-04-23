import json
import pathlib
from app.training.config_summary import Config_Summary


class ConfigLoader:
    def __init__(self, config_file: pathlib.Path):
        with open(config_file, 'r') as file:
            self.config_data = json.load(file)
        self.config = Config_Summary(**self.config_data)

    def get_config(self) -> Config_Summary:
        return self.config

    def get_raw_config(self) -> dict:
        return self.config_data
