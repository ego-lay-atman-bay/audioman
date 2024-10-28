from .effect import Effect

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio import Audio
    
class SetVolume(Effect):
    TYPE = 'audio'
    OPTIONS = {
        'db': {
            'default': -20,
            'type': float,
        }
    }
    DEFAULT = 'db'
    
    def apply(self, samples: 'Audio') -> 'Audio':
        return self.match_target_amplitude(samples, self.options['db'])
    
    def match_target_amplitude(self, sound: 'Audio', target_dBFS: float) -> 'Audio':
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
