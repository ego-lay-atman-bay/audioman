from typing import TYPE_CHECKING

from pydub import AudioSegment
from pydub.effects import normalize

from .effect import Effect

if TYPE_CHECKING:
    from ..audio import Audio
    
class Normalize(Effect):
    OPTIONS = {
        "headroom": {
            "type": float,
            "default": 0.1,
        }
    }
    TYPE = "audio"
    DEFAULT = "headroom"
    
    def apply(self, samples: 'Audio') -> 'Audio':
        with samples.pydub() as audio:
            audio.set(normalize(audio.audio_segment, self.options['headroom']))
        
        return samples
