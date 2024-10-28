import typing

from .effect import Effect
# To prevent circular import when running
if typing.TYPE_CHECKING:
    from ..audio import Audio
    
def detect_leading_silence(sound: 'Audio', silence_threshold = -50.0, chunk_size = 10):
    """
    Taken from pydub
    
    Returns the millisecond/index that the leading silence ends.

    audio_segment - the segment to find silence in
    silence_threshold - the upper bound for how quiet is silent in dFBS
    chunk_size - chunk size for interating over the segment in ms
    """
    trim_samples = 0 # samples
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_samples:trim_samples + chunk_size].dBFS < silence_threshold and trim_samples < len(sound):
        trim_samples += chunk_size

    # if there is no end it should return the length of the segment
    return min(trim_samples, len(sound))

class TruncateSilence(Effect):
    OPTIONS = {
        'threshold': {
            'default': -50.0,
            'type': float,
        },
        'chunk_size': {
            'default': 0.01,
            'type': float,
        },
        'keep_start': {
            'default': None,
            'type': float,
        },
        'keep_end': {
            'default': None,
            'type': float,
        },
        'keep': {
            'default': 0,
            'type': float,
        }
    }
    TYPE = 'audio'
    
    def __init__(self, options: dict = None, **kwargs) -> None:
        super().__init__(options, **kwargs)
        
        if self.options['keep_start'] == None:
            self.options['keep_start'] = self.options['keep']
        
        if self.options['keep_end'] == None:
            self.options['keep_end'] = self.options['keep']
    
    def apply(self, samples: 'Audio') -> 'Audio':
        # with samples.pydub() as audio:
        return self.strip_silence(samples)
        
        return samples
    
    def trim_leading_silence(self, audio: 'Audio', keep: int | float = 0) -> 'Audio':
        silent_start = detect_leading_silence(
            audio,
            silence_threshold = self.options['threshold'],
            chunk_size = audio.seconds_to_samples(self.options['chunk_size']),
        )
        
        keep = audio.seconds_to_samples(keep)
        
        adjusted_time = silent_start - keep
        if adjusted_time < 0:
            audio = audio.add_silence(keep - silent_start)
            adjusted_time +=  keep - silent_start
        
        return audio[adjusted_time:]
    
    def trim_trailing_silence(self, audio: 'Audio', keep: int | float = 0):
        return self.trim_leading_silence(audio.reverse(), keep = keep).reverse()
    
    def strip_silence(self, audio: 'Audio'):
        return self.trim_trailing_silence(self.trim_leading_silence(audio, keep = self.options['keep_start']), keep = self.options['keep_end'])
