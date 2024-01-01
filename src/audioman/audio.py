import typing

import logging
import os
from copy import deepcopy
import numpy
import soundfile
import tempfile

from .effect import Effect
from .audio_tags import AudioTags

class Audio:
    def __init__(self, file: str | numpy.ndarray = numpy.array([]), sample_rate: int = None) -> None:
        """Audio object. This is used to manipulate audio samples. Please note, `self.filename` is usually lost when editing audio.

        Args:
            file (str | numpy.ndarray, optional): File to load, or samples as numpy array. Defaults to numpy.array([]).
            sample_rate (int, optional): Sample rate. Defaults to file sample rate or 44000.
        """
        self._samples: numpy.ndarray = None
        self.filename: str = ''
        
        self.tags = AudioTags()
        
        if isinstance(file, str):
            self.filename = file
            self.read()
            if sample_rate != None:
                self.sample_rate = sample_rate
        elif isinstance(file, numpy.ndarray):
            if sample_rate == None:
                sample_rate = 44000
                
            self.sample_rate = sample_rate
            
            self.samples: numpy.ndarray = file.copy()
    
    @property
    def cache_filename(self) -> str:
        """The cache filename to use when using the `.unload()` method.

        Returns:
            str: filename
        """
        if not hasattr(self, '_cache_filename') or not os.path.exists(self._cache_filename):
            
            with tempfile.NamedTemporaryFile(
                'w',
                prefix = 'audioman_',
                suffix = '.wav',
                delete = False,
            ) as file:
                file.write('')
            
            self._cache_filename = file.name
        
        return self._cache_filename

    def __del__(self):
        if hasattr(self, '_cache_filename') and os.path.exists(self._cache_filename):
            os.remove(self._cache_filename)
            logging.debug(f'deleted: {self._cache_filename}')
            
    def unload(self):
        """Unload audio samples to save memory. This will also create a temporary wav file in the os temp directory. When you try to access the samples, this temporary file will be loaded (or at least it will try to load it).
        """
        filename = self.filename
        try:
            self.save(self.cache_filename)
            self.samples = None
        except:
            logging.debug(f'cannot save file {self.cache_filename}', exc_info = True)
        self.filename = filename
    
    @property
    def samples(self) -> numpy.ndarray:
        """Audio samples as numpy array.

        Returns:
            numpy.ndarray: Numpy array. Shape is (channels, length)
        """
        if self._samples is None:
            filename = self.filename
            try:
                self.read(self.cache_filename)
            except:
                self.read(filename)
            
            self.filename = filename
        
        return self._samples
    @samples.setter
    def samples(self, value: numpy.ndarray):
        self._samples = value
        
    def read(self, filename: str = None):
        """Read audio file.

        Args:
            filename (str, optional): File to read. Defaults to `self.filename`.

        Raises:
            FileNotFoundError: file not found
            IsADirectoryError: path specified is a directory
        """
        if not filename == None:
            self.filename = filename
        if filename == None:
            filename = self.filename
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"file '{filename}' not found")
        if os.path.isdir(filename):
            raise IsADirectoryError(f"path '{filename}' is a directory, not a file")
        
        audio, self.sample_rate = soundfile.read(filename, always_2d = True)
        self.samples = audio.swapaxes(1,0)
        self.tags.load(filename)
    
    def save(self, filename: str = None, format = None):
        """Save file. If the filename is specified, it override `.filename` attribute.

        Args:
            filename (str, optional): File to save audio to. Defaults to `self.filename`.
            format (str, optional): File format to save the audio to. Defaults to None.

        Raises:
            TypeError: filename must be str
        """
        if filename == None:
            filename = self.filename
        
        if not isinstance(filename, str):
            raise TypeError('filename must be str')
        
        if filename != None:
            self.filename = filename
        
        if isinstance(self.filename, str):
            soundfile.write(self.filename, self.samples.swapaxes(1, 0), samplerate = self.sample_rate, format = format)
            self.tags.save(self.filename)
    
    @property
    def channels(self) -> int:
        """Number of audio channels.

        Returns:
            int: Number of audio channels.
        """
        return self.samples.shape[0]
    @channels.setter
    def channels(self, channels: int):
        """Set number of channels. If new number of channels is 1, all channels will be combined to mono. If the number of channels is smaller than current channels, then audio will be lost. If new number of channels is greater than current number of channels, then it will add new channels from current channels.

        Args:
            channels (int): Number of new channels.
        """
        channels = int(channels)
        
        if channels <= 0:
            raise ValueError('number of channels must be at least 1')
        
        samples = self.samples.copy()
        if channels > self.channels:
            for channel in range(channels - self.channels):
                samples = numpy.append(samples, [self.samples[(channel + self.channels) % self.channels]], axis = 0)
        elif channels == 1:
            samples = sum(samples)
        elif channels < self.channels:
            samples = self.samples[:channels]
        else:
            return
        
        self.samples = samples
    
    def seconds_to_samples(self, duration: float, sample_rate: int = None) -> int:
        """Convert seconds to samples.

        Args:
            duration (float): Duration in seconds.
            sample_rate (int, optional): Sample rate. Defaults to `self.sample_rate`.

        Returns:
            int: duration in samples
        """
        
        if sample_rate == None:
            sample_rate = self.sample_rate
        
        return int(duration * sample_rate)
    
    def samples_to_seconds(self, duration: int, sample_rate: int = None) -> float:
        """Converts samples to seconds.

        Args:
            duration (int): Duration in samples
            sample_rate (int, optional): Sample rate. Defaults to `self.sample_rate`.

        Returns:
            float: Seconds.
        """
        
        if sample_rate == None:
            sample_rate = self.sample_rate
            
        return duration / sample_rate
    
    def add_silence(self, start: int = 0, length: int = None) -> 'Audio':
        """Add silence to audio.

        Args:
            start (int, optional): Start sample to start the audio at. If it's less than -1, then it will start at the end. Defaults to 0.
            length (int, optional): Silence length in samples. Defaults to None.

        Returns:
            Audio: New Audio with silence.
        """
        if length == None:
            length = start
            start = 0
        
        if start < 0:
            if start == -1:
                beginning = self.samples.copy()
                end = numpy.array([[]] * self.channels)
            else:
                beginning, end = numpy.split(self.samples, [start + 1], axis = 1)
        else:
            beginning, end = numpy.split(self.samples, [start], axis = 1)
        
        middle = numpy.array([[0] * length] * self.channels)

        samples = numpy.append(numpy.append(beginning, middle, axis = 1), end, axis = 1)
        
        audio = Audio(samples, sample_rate = self.sample_rate)
        audio.filename = self.filename
        
        return audio
    
    def split(self, middle: int = None):
        """Split audio by middle sample

        Args:
            middle (int, optional): Middle sample to split by. Defaults to half.

        Returns:
            tuple[Audio,Audio]: Split audio.
        """
        if middle == None:
            middle = self.length // 2
        
        if middle < 0:
            if middle == -1:
                beginning = self.samples.copy()
                end = numpy.array([[]] * self.channels)
            else:
                beginning, end = numpy.split(self.samples, [middle + 1], axis = 1)
        else:
            beginning, end = numpy.split(self.samples, [middle], axis = 1)
        
        beginning_audio = Audio(beginning, self.sample_rate)
        end_audio = Audio(end, self.sample_rate)
        
        return (beginning_audio, end_audio)
    
    def trim(self, start: int = 0, length: int = None):
        """Trim audio to specified length from start.

        Args:
            start (int, optional): Start in samples. Defaults to 0.
            length (int, optional): Length in samples. If not specified, the start will act as the length. Defaults to None.

        Returns:
            Audio: trimmed audio.
        """
        if length == None:
            length = start
            start = 0
            
        if start < 0:
            start += 1
            
            if start == 0:
                return self.copy()
            start = self.length + start
        
        end = min(start + length, self.length)
        
        samples = self.samples.swapaxes(1,0)[start:end].swapaxes(1,0)
        return Audio(samples, sample_rate = self.sample_rate)
    
    def apply_effect(self, effect: Effect, start: int = 0):
        """Apply effect.

        Args:
            effect (Effect): Effect to apply to audio. This effect must be an object that inherits from the `Effect` class.
            start (int, optional): Where to start the effect in the audio in samples. Defaults to 0.

        Returns:
            Audio: New Audio with applied effect.

        Raises:
            TypeError: effects must inherit from the Effect class
        """
        if not isinstance(effect, Effect):
            raise TypeError('effects must inherit from the Effect class')
        
        if not isinstance(effect.length, int):
            if isinstance(effect.length, (float, str)):
                effect.length = int(effect.length)
            else:
                effect.length = self.length
        
        return self.apply_scaler(effect.get(), start = start)
    
    def apply_scaler(self, scaler: numpy.ndarray, start: int = 0) -> 'Audio':
        """Apply a scaler to the audio samples.

        Args:
            scaler (numpy.ndarray): Scaler to apply to the audio samples.
            start (int, optional): Where to apply the scaler in samples. Defaults to 0.
        
        Returns:
            Audio: New Audio with applied scaler
        """
        samples = self.samples.copy()

        length = len(scaler)
        
        if start < 0:
            # start += 1
            start = self.length + start
        
        end = min(start + length, self.length)
        scaler = scaler[:end - start]

        for channel in samples:
            channel[start:end] = channel[start:end] * scaler
        
        audio = Audio(samples, sample_rate = self.sample_rate)
        audio.filename = self.filename
        
        return audio
    
    def mix(self, audio2: 'Audio') -> 'Audio':
        """Mix audio together. This will overlay this audio and the new audio onto each other, so they play at the same time.

        Args:
            audio2 (Audio): Audio to mix into this audio.

        Returns:
            Audio: New mixed audio.
        """

        audio1 = self.copy()
        audio2 = audio2.copy()

        if audio1.channels > audio2.channels:
            audio2.channels = audio1.channels
        elif audio2.channels > audio1.channels:
            audio1.channels = audio2.channels
        
        if audio1.length > audio2.length:
            audio2.add_silence(-1, audio1.length - audio2.length)
        elif audio2.length > audio1.length:
            audio1.add_silence(-1, audio2.length - audio1.length)
        
        samples = (audio1.samples + audio2.samples).clip(-1.0,1.0)
        audio = Audio(samples, audio1.sample_rate)
        return audio
    
    def __add__(self, value: int | float):
        if not isinstance(value, (int, float, Audio, numpy.ndarray)):
            raise TypeError(f"unsupported operand type(s) for +: 'Audio' and '{value.__class__.__qualname__}'")

        if isinstance(value, Audio):
            if value.samples.shape[0] != self.samples.shape[0]:
                raise TypeError('cannot add audio with different number of channels')
            
            samples = numpy.append(self.samples, value.samples, axis = 1)
            
            audio = Audio(samples, sample_rate = self.sample_rate)
            return audio
        
        elif isinstance(value, numpy.ndarray):
            if value.shape[0] != self.samples.shape[0]:
                raise TypeError('cannot add audio with different number of channels')
            
            samples = numpy.append(self.samples, value, axis = 1)
            
            audio = Audio(samples, sample_rate = self.sample_rate)
            return audio
        
        elif isinstance(value, (int, float)):
            samples = self.samples + value
            
            audio = Audio(samples, sample_rate = self.sample_rate)
            return audio
    
    def __radd__(self, value: int | float):
        return self.__add__(value)
    
    def __iadd__(self, value: float | int):
        audio = self.__add__(value)
        self.samples = audio.samples
        return self
    
    def __sub__(self, value: float | int):
        if isinstance(value, Audio):
            raise TypeError('cannot subtract Audio from each other')
        if isinstance(value, (float, int)):
            samples = self.samples - value
            return Audio(samples, sample_rate = self.sample_rate)
    
    def __rsub__(self, value: float | int):
        return self.__sub__(value)
    
    def convert_to_sample_rate(self, sample_rate: int):
        """I want this to be able to convert a sound to a different sample rate without changing how it sounds. However, right now it just sets the sample rate without changing the samples.

        Args:
            sample_rate (int): new sample rate

        Raises:
            TypeError: sample_rate must be 'int'
            ValueError: sample_rate must be greater than 0
        """
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be 'int'")
        if sample_rate < 1:
            raise ValueError("sample_rate must be greater than 0")
        
        self.sample_rate = sample_rate
    
    @property
    def length(self) -> int:
        """Length of audio in samples.

        Returns:
            int: Number of samples.
        """
        return self.__len__()
    
    def __len__(self) -> int:
        return self.samples.shape[1]
    
    def copy(self) -> 'Audio':
        """Copy this audio into a new Audio object.

        Returns:
            Audio: New Audio object.
        """
        audio = Audio(self.samples, self.sample_rate)
        audio.filename = self.filename
        audio.tags = deepcopy(self.tags)
        return audio
