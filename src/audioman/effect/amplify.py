import numpy

from .effect import Effect


class Amplify(Effect):
    """Amplify audio.

    Options:
        gain (float): Amount to amplify by. Must be scaler value.
    """
    OPTIONS = {
        "gain": {
            "default": 0,
            "type": float,
        }
    }
    DEFAULT = "gain"
    
    def apply(self, samples: numpy.ndarray):
        return samples * (10 ** (self.options['gain'] / 20))
