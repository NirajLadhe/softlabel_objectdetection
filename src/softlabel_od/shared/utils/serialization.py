"""JSON serialization helpers shared across the package."""

import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is None:
            return None
        return super().default(obj)
