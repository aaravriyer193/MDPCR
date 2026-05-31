# MDPCR — utils/logging.py
import os
import json
from datetime import datetime
import config


class TrainingLogger:
    def __init__(self, path: str = config.LOG_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path    = path
        self.history = []

    def log(self, **kwargs) -> None:
        entry = {"timestamp": datetime.now().isoformat(), **kwargs}
        self.history.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
