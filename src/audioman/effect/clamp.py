from numpy import ndarray

from .effect import Effect


class Clamp(Effect):
    TYPE = 'samples'
    OPTIONS = {
        'threshold': {
            'type': float,
            'default': 1.0,
        }
    }
    DEFAULT = 'threshold'
    
    def apply(self, samples: ndarray) -> ndarray:
        maximum = max(abs(samples.max()), abs(samples.min()))

        if maximum > self.options['threshold']:
            samples = samples * (self.options['threshold'] / maximum)

        return samples
