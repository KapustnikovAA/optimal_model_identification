import yaml
from typing import Dict

def load_yaml (config_path: str) -> Dict:

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config