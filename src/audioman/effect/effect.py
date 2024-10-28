from abc import abstractmethod
import typing
from typing import Literal, Any

import numpy

# To prevent circular import when running
if typing.TYPE_CHECKING:
    from ..audio import Audio

class Effect:
    OPTIONS: dict[str, dict[Literal['default', 'type'], Any]] = {}
    DEFAULT: str = None
    TYPE: Literal['samples', 'audio'] = 'samples'
    
    def __init__(
        self,
        options: dict = None,
        **kwargs,
    ) -> None:
        """Audio effect. This effect will last the specified length in samples. If the duration is specified (in seconds), then it will use that length. This will assume the sample rate is `4400`, unless specified.

        Args:
            **options (dict[str, Any]): Effect options. Not all effects have the same options. Available options can be found in the `OPTIONS` property.
        """

        self.options = {}
        
        if options == None:
            options = {}

        for option in self.OPTIONS:
            self.options[option] = self.OPTIONS[option]['default']
        
        for option in kwargs:
            if kwargs[option] == None:
                self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
                continue
            if option in self.OPTIONS and 'type' in self.OPTIONS[option]:
                try:
                    self.options[option] = self.OPTIONS[option]['type'](kwargs[option])
                except:
                    self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
            else:
                self.options[option] = kwargs[option]
            
        for option in options:
            if option in self.OPTIONS and 'type' in self.OPTIONS[option]:
                if options[option] == None:
                    self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
                    continue
                try:
                    self.options[option] = self.OPTIONS[option]['type'](options[option])
                except Exception as e:
                    e.add_note(f'{option}: {options[option]}')
                    self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
                    raise e
            else:
                self.options[option] = options[option]
    
    def validate_options(self, options: dict[str, Any]):
        for option in options:
            if options[option] == None:
                self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
                continue
            if option in self.OPTIONS and 'type' in self.OPTIONS[option]:
                try:
                    self.options[option] = self.OPTIONS[option]['type'](options[option])
                except:
                    self.options[option] = self.OPTIONS[option].get('default', self.OPTIONS[option]['type']())
            else:
                self.options[option] = options[option]
    
    @abstractmethod
    def apply(self, samples: 'numpy.ndarray | Audio') -> 'numpy.ndarray | Audio':
        pass
    
    
