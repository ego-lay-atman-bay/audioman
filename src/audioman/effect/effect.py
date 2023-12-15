from typing import Literal, Any

import numpy
from copy import deepcopy

class Effect:
    OPTIONS: dict[str, dict[Literal['default', 'type'], Any]] = {}
    
    def __init__(
        self,
        length: int = None,
        duration: float = None,
        sample_rate: int = None,
        options: dict = {},
        **kwargs,
    ) -> None:
        """Audio effect. This effect will last the specified length in samples. If the duration is specified (in seconds), then it will use that length. This will assume the sample rate is `4400`, unless specified.

        Args:
            length (int, optional): Length of effect in samples. Defaults to None.
            duration (float, optional): Duration of effect in seconds. Defaults to None.
            sample_rate (int, optional): Audio Sample rate. Defaults to 4400. This may be used by some effects.
            **options (dict[str, Any]): Effect options. Not all effects have the same options. Available options can be found in the `OPTIONS` property.
        """
        self.sample_rate = 4400
        if isinstance(sample_rate, (int, float)):
            sample_rate = int(sample_rate)
            self.sample_rate = sample_rate
            
        self.length = 0
        if isinstance(length, int):
            self.length = length
        elif isinstance(duration, (int, float)):
            self.length = int(duration * self.sample_rate)
        

        self.options = {}

        for option in self.OPTIONS:
            self.options[option] = self.OPTIONS[option]['default']
        
        for option in kwargs:
            if option in self.OPTIONS and 'type' in self.OPTIONS[option]:
                self.options[option] = self.OPTIONS[option]['type'](kwargs[option])
            else:
                self.options[option] = kwargs[option]
            
        for option in options:
            if option in self.OPTIONS and 'type' in self.OPTIONS[option]:
                self.options[option] = self.OPTIONS[option]['type'](options[option])
            else:
                self.options[option] = options[option]
    
    def get(self) -> numpy.ndarray:
        """Get the effect scaler array.

        Returns:
            numpy.ndarray: 1-d numpy array with values between 0 and 1.
        """
        return numpy.array([1] * self.length)
