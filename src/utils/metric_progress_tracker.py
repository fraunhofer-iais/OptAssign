import json
import os


class MetricProgressTracker:

    def __init__(self, path: str):
        self.path = path
        with open(self.path, "w") as f:
            pass

    def track(self, _data: dict):
        with open(self.path, "r+") as f:
            try:
                data = json.load(f)
            except:
                data = []

            data.append(_data)
        os.remove(self.path)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)
