import json
from pathlib import Path
from itertools import cycle
from app.training.config_summary import Config_Summary


# Take a path to prompt file and load these prompts to apply augmentation with equally distributred prompts
def load_prompts_in_order(config=Config_Summary):
    prompt_path = Path(config.prompt_dir)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Path for prompts: {prompt_path} not found.")

    with open(prompt_path, "r") as f:
        prompts = json.load(f)["prompts"]
    return cycle(prompts)
