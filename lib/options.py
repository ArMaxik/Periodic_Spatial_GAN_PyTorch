import json

class options:
    def __init__(self, config: str):
        with open(config, 'r') as f:
            config_j = json.loads(f.read())
        for k, v in config_j.items():
            setattr(self, k, v)
