import array
import logging
import math
import os
import tempfile
from copy import deepcopy
from typing import Annotated, overload, TYPE_CHECKING

import numpy
import pydub.effects
import soundfile

from .audio_tags import AudioTags
from .effect import Effect
from .ffmpeg_commands import log_ffmpeg_output, setup_ffmpeg_log, test_ffmpeg, run_ffmpeg_command
from .format_str import format_str
from .utils import ratio_to_db

if TYPE_CHECKING:
    import pydub

class Audio:
    pydub_sample_width: int = 2
    
    def __init__(self, file: str | numpy.ndarray = None, sample_rate: int = None) -> None:
        """Audio object. This is used to manipulate audio samples. Please note, `self.filename` is usually lost when editing audio.

        Args:
            file (str | numpy.ndarray, optional): File to load, or samples as numpy array. Defaults to numpy.array([]).
            sample_rate (int, optional): Sample rate. Defaults to file sample rate or 44000.
        """
        self.__raw_cache = True
        
        self._samples: numpy.ndarray = None
        self.filename: str = ''
        
        self.tags = AudioTags()
        
        if file is None:
            file = numpy.array([])
        
        if isinstance(file, str):
            self.filename = file
            self.read()
            if sample_rate != None:
                self.sample_rate = sample_rate
        elif isinstance(file, (numpy.ndarray, list)):
            if sample_rate == None:
                sample_rate = 44000
                
            self.sample_rate = sample_rate
            
            if isinstance(file, numpy.ndarray):
                self.samples: numpy.ndarray = file.copy()
            elif isinstance(file, list):
                self.samples: numpy.ndarray = numpy.array(file)
    
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
    
    @property
    def raw_cache_filename(self) -> str:
        """The raw data cache filename to use when using the `.unload()` method.

        Returns:
            str: filename
        """
        if not hasattr(self, '_raw_cache_filename') or not os.path.exists(self._raw_cache_filename):
            
            with tempfile.NamedTemporaryFile(
                'w',
                prefix = 'audioman_',
                suffix = '.npy',
                delete = False,
            ) as file:
                file.write('')
            
            self._raw_cache_filename = file.name
        
        return self._raw_cache_filename

    def __del__(self):
        if hasattr(self, '_cache_filename') and os.path.exists(self._cache_filename):
            os.remove(self._cache_filename)
            logging.debug(f'deleted: {self._cache_filename}')
        if hasattr(self, '_raw_cache_filename') and os.path.exists(self._raw_cache_filename):
            os.remove(self._raw_cache_filename)
            logging.debug(f'deleted: {self._raw_cache_filename}')
            
    def unload(self, raw = True):
        """Unload audio samples to save memory. This will also create a temporary wav file in the os temp directory. When you try to access the samples, this temporary file will be loaded (or at least it will try to load it).
        
        This method will return the filename. This can be used to edit the file in an external editor, save it to the same file, and then load it again.

        Args:
            raw (bool, optional): save the samples as a raw .npy file instead of a .wav file.
        
        Returns:
            str: Temporary filename.
        """
        filename = self.filename
        
        if raw:
            try:
                numpy.save(self.raw_cache_filename, self.samples)
                self.samples = None
            except:
                logging.debug(f'cannot save file {self.raw_cache_filename}', exc_info = True)
        else:
            try:
                self.save(self.cache_filename)
                self.samples = None
            except:
                logging.debug(f'cannot save file {self.cache_filename}', exc_info = True)

        self.__raw_cache = raw
        
        return self.raw_cache_filename if raw else self.cache_filename
    
    @property
    def samples(self) -> numpy.ndarray:
        """Audio samples as numpy array.

        Returns:
            numpy.ndarray: Numpy array. Shape is (channels, length)
        """
        if self._samples is None:
            filename = self.filename
            try:
                if self.__raw_cache:
                    self.samples = numpy.load(self.raw_cache_filename)
                else:
                    self.read(self.cache_filename)
            except:
                self.read(filename)
        
        return self._samples
    @samples.setter
    def samples(self, value: numpy.ndarray):
        self._samples = value
    
    @property
    def rms(self):
        return math.sqrt(numpy.mean(self.samples**2))
        
    @property
    def dBFS(self):
        return ratio_to_db(self.rms)
    
    def read(self, filename: str | None = None):
        """Read audio file.

        Args:
            filename (str, optional): File to read. Defaults to `self.filename`.

        Raises:
            FileNotFoundError: file not found
            IsADirectoryError: path specified is a directory
        """
        if filename not in [self.cache_filename, self.raw_cache_filename] and not filename == None:
            self.filename = filename
        if filename == None:
            filename = self.filename
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"file '{filename}' not found")
        if os.path.isdir(filename):
            raise IsADirectoryError(f"path '{filename}' is a directory, not a file")
        
        if filename not in [self.cache_filename, self.raw_cache_filename]:
            command = f'-hide_banner -loglevel error -y -i "{filename}" "{self.cache_filename}"'
            
            run_ffmpeg_command(command)
        
        audio, self.sample_rate = soundfile.read(self.cache_filename, always_2d = True)
        self.samples = audio.swapaxes(1,0)
        
        if filename not in [self.cache_filename, self.raw_cache_filename]:
            self.tags.load(filename)
    
    def save(self, filename: str | None = None, file_format: str | None = None, ffmpeg_options: str | None = None):
        """Save file. If the filename is specified, it override `.filename` attribute.

        Args:
            filename (str, optional): File to save audio to. Defaults to `self.filename`.
            file_format (str, optional): File format to save the audio to. Defaults to None.
            ffmpeg_options (str, optional): Custom ffmpeg export options. Defaults to '-i "{input}" "{output}"'.

        Raises:
            TypeError: filename must be str
        
        ## Custom ffmpeg options
        This is the ffmpeg export command. The input will be formatted with the input, output, format, and metadata. It will use the standard python formatting.
        
        Custom ffmpeg options can be used to add compression, custom codecs, and processing that is not easily done using just the samples.
        
        Examples:
        ```
        -i "{input}" "{output}"
        ```
        ```
        -i "{input}" -acodec flac -compression_level 12 "{output}"
        ```
        ```
        -i "{input}" -compression_level 12 "{output}"
        ```
        
        It can even be used to add an image to audio and make a video from it.
        `-loop 1 -i "img.jpg" -i "{input}" -shortest "{output_name}.{extension}"`
        
        **Make sure to use `{input}` and `{output}` or `{output_name}.{extension}` as `output` is used when saving tags.**
        
        Please note: ffmpeg options will include `-hide_banner`, and `-y`. Files will be replaced without confirmation.
        """
        
        if filename == None:
            filename = self.filename
        
        if not isinstance(filename, str):
            raise TypeError('filename must be str')
        
        if filename not in [self.cache_filename, self.raw_cache_filename] and filename != None:
            self.filename = filename
            
        if file_format == None:
            file_format = os.path.splitext(filename)[1][1::]
        
        if ffmpeg_options is None:
            try:
                if isinstance(filename, str):
                    soundfile.write(filename, self.samples.swapaxes(1, 0), samplerate = self.sample_rate, format = file_format)
                    self.tags.save(filename)
            except:
                ffmpeg_options = ['-i', self.cache_filename, filename]

        if ffmpeg_options is not None:
            if not isinstance(ffmpeg_options, (str, list, tuple)):
                raise TypeError('ffmpeg options must be "str", "list", or "tuple')
            
            command = ['-hide_banner', '-y', '-loglevel', 'error']
            
            if isinstance(ffmpeg_options, (list, tuple)):
                command += ffmpeg_options
            else:
                command = f"{' '.join(command)} {ffmpeg_options}"
            
            soundfile.write(self.cache_filename, self.samples.swapaxes(1, 0), samplerate = self.sample_rate)
            
            
            format_options = self.tags.expand()
            format_options['input'] = self.cache_filename
            format_options['output'] = filename
            format_options['output_name'] = os.path.splitext(filename)[0]
            format_options['output_folder'] = os.path.dirname(filename)
            format_options['extension'] = os.path.splitext(filename)[1][1::]
            format_options['format'] = file_format
            
            
            if isinstance(command, list):
                for i in range(len(command)):
                    command[i] = format_str(str(command[i]), **format_options)
            elif isinstance(command, str):
                command = format_str(str(command), **format_options)
            
            run_ffmpeg_command(command)

            try:
                self.tags.save(filename)
            except:
                logging.info(f'cannot add tags to "{filename}"')
                logging.log(msg = 'exception', level = logging.DEBUG, exc_info = True)
        
    
    
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

    def samples_to_milliseconds(self, duration: int, sample_rate: int = None) -> float:
        """Converts samples to seconds.

        Args:
            duration (int): Duration in samples
            sample_rate (int, optional): Sample rate. Defaults to `self.sample_rate`.

        Returns:
            float: Seconds.
        """
        
        if sample_rate == None:
            sample_rate = self.sample_rate
            
        return (duration * 1000) / sample_rate
    
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
        
        audio = self.copy(samples)
        
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
            
            start = self.length + start
        
        if length < 0:
            length = (self.length - start) + (length + 1)
        
        end = min(start + length, self.length)
        
        samples = self.samples[:,start:end]
        return self.copy(samples)
    
    def __getitem__(self, key: int | float | slice):
        if isinstance(key, tuple):
            return tuple(self[k] for k in key)
        if isinstance(key, float):
            key = self.seconds_to_samples(key)
        if isinstance(key, int):
            return self.samples[:,key]
        if isinstance(key, slice):
            start = key.start
            step = key.step
            stop = key.stop
            
            if start == None:
                start = 0
            if stop == None:
                stop = -1
            
            
            if isinstance(start, float) or isinstance(stop, float):
                start = self.seconds_to_samples(start)
                stop = self.seconds_to_samples(stop)
            
            if start < 0:
                start += 1
                
                start = self.length + start
            
            if stop < 0:
                stop = self.length + (stop + 1)
                
            flipped = False
            
            if step != None and step < 0:
                flipped = True
                step = abs(step)
            
            samples = self.samples[:,start:stop:step]
            
            if flipped:
                samples = numpy.flip(samples, 1)
            
            return self.copy(samples)

    
    def apply_effect(self, effect: Effect, start: int = None, length: int | None = None):
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
        
        split = self.split_around(start, length)
        
        if effect.TYPE == 'audio':
            middle = Audio(split[1], self.sample_rate)
        else:
            middle = split[1]
            
        applied = effect.apply(middle)
        
        new_samples = None
        
        if isinstance(applied, Audio):
            new_samples = applied.samples
        elif isinstance(applied, numpy.ndarray):
            new_samples = applied
        else:
            raise TypeError('effect returned non Audio or numpy.ndarray')
        
        logging.debug(f'shape: {self.samples.shape}')
        logging.debug(f'split[0]: {repr(split[0])}')
        logging.debug(f'new_samples: {repr(new_samples)}')
        logging.debug(f'split[2]: {repr(split[2])}')
        
        return self.copy(
            numpy.concatenate(
                (
                    split[0],
                    new_samples,
                    split[2],
                ),
                axis = 1,
            )
        )
    
    @overload
    def split_around(self) -> Annotated[list[numpy.ndarray], 3]: ...
    @overload
    def split_around(self, length: int) -> Annotated[list[numpy.ndarray], 3]: ...
    @overload
    def split_around(self, start: int, length: int) -> Annotated[list[numpy.ndarray], 3]: ...
    def split_around(self, start: int | None = None, length: int | None = None) -> Annotated[list[numpy.ndarray], 3]:
        """Split samples from start to length, keeping all parts.

        Args:
            start (int, optional): Where to apply the scaler in samples. Defaults to 0.
            length (int, optional): The length to split by. Defaults to length of audio.
        
        Returns:
            list[numpy.ndarray]: [start, middle, end]
        """
        samples = self.samples.copy()

        if length == None:
            if (start == None):
                length = self.length
            else:
                length = start
            start = 0
        
        if start == None:
            start = 0
        
        if start < 0:
            start += 1
            
            start = self.length + start
        
        if length < 0:
            length = (self.length - start) + (length + 1)
        
        end = min(start + length, self.length)

        return [
            samples[:,0:max(0, start)],
            samples[:,start:end],
            samples[:,min(end, self.length):-1]
        ]
    
    def mix(self, audio2: 'Audio', start: int = 0) -> 'Audio':
        """Mix audio together. This will overlay this audio and the new audio onto each other, so they play at the same time.

        Args:
            audio2 (Audio): Audio to mix into this audio.
            start (int, optional): Second audio start sample. Defaults to 0.

        Returns:
            Audio: New mixed audio.
        """

        audio1 = self.copy()
        audio2 = audio2.copy()
        
        audio2 = audio2.add_silence(0 if start >= 0 else -1, abs(start))

        if audio1.channels > audio2.channels:
            audio2.channels = audio1.channels
        elif audio2.channels > audio1.channels:
            audio1.channels = audio2.channels
        
        if audio1.length > audio2.length:
            audio2 = audio2.add_silence(-1, audio1.length - audio2.length)
        elif audio2.length > audio1.length:
            audio1 = audio1.add_silence(-1, audio2.length - audio1.length)
        
        samples = (audio1.samples + audio2.samples).clip(-1.0,1.0)
        audio = self.copy(samples, audio1.sample_rate)
        return audio
    
    def reverse(self):
        return self[::-1]
    
    def apply_gain(self, gain: float = 0) -> 'Audio':
        return self.copy(self.samples * (10 ** (gain / 20)))
    
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
            return self.apply_gain(value)
    
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
            return self.apply_gain(-value)
    
    def __rsub__(self, value: float | int):
        return self.__sub__(value)
    
    def set_sample_rate(self, sample_rate: int):
        """I want this to be able to convert a sound to a different sample rate without changing how it sounds. However, right now it just sets the sample rate without changing the samples.

        Args:
            sample_rate (int): new sample rate

        Raises:
            TypeError: sample_rate must be 'int'
            ValueError: sample_rate must be greater than 0
        """
        assert isinstance(sample_rate, int)
        assert sample_rate > 1
        
        
        self.sample_rate = sample_rate
    
    def copy(
        self,
        samples: numpy.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> "Audio":
        """Create a copy of this Audio object, with the option of passing in custom samples or sample rate. This is useful for keeping the filename on a modified Audio object.

        Args:
            samples (numpy.ndarray | None, optional): New samples. Defaults to current samples.
            sample_rate (int | None, optional): New sample rate. Defaults to current sample rate.

        Returns:
            Audio: New Audio object with the same filename.
        """
        if samples is None:
            samples = self.samples.copy()
        if sample_rate == None:
            sample_rate = self.sample_rate
        
        audio = Audio(
            samples,
            sample_rate,
        )
        
        audio.filename = self.filename
        audio.tags = deepcopy(self.tags)
        
        return audio
    
    @property
    def length(self) -> int:
        """Length of audio in samples.

        Returns:
            int: Number of samples.
        """
        return self.__len__()
    
    def __len__(self) -> int:
        return self.samples.shape[1]
    
    def pydub(self):
        return PyDubAudioSegment(self)
    
class PyDubAudioSegment():
    def __init__(self, audio: Audio) -> None:
        import pydub
        
        if not isinstance(audio, Audio):
            raise TypeError('value must be Audio object')
        
        self._audio: Audio = audio
        self.audio_segment: 'pydub.AudioSegment' = None
        self.filename: str = None
    
    def __enter__(self):
        self.audio_segment = self.np_to_pydub(self._audio.samples, self._audio.sample_rate)
        self.filename = self._audio.unload()
        # self.audio_segment = pydub.AudioSegment.from_file(self.filename)
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        samples, sample_rate = self.pydub_to_np(self.audio_segment)
        self._audio.samples = samples
        self._audio.sample_rate = sample_rate
        
        return False
    
    def set(self, segment: 'pydub.AudioSegment'):
        import pydub
        
        if not isinstance(segment, pydub.AudioSegment):
            raise TypeError('value must be pydub AudioSegment')
        
        self.audio_segment = segment
    
    def np_to_pydub(self, samples: numpy.ndarray, sample_rate: int):
        import pydub

        new = samples.swapaxes(1,0).reshape(-1) * (1 << (8 * self._audio.pydub_sample_width - 1))

        sample_array = array.array('l', new.astype(numpy.int64))

        return pydub.AudioSegment(sample_array, frame_rate = sample_rate, channels = samples.shape[0], sample_width = self._audio.pydub_sample_width)
    
    def pydub_to_np(self, audio: 'pydub.AudioSegment') -> tuple[numpy.ndarray, int]:
        """
        This was taken from https://stackoverflow.com/a/66922265/17129659
        
        Converts pydub audio segment into numpy.float64 of shape [channels, duration_in_seconds*sample_rate],
        where each value is in range [-1.0, 1.0]. 
        Returns tuple (audio_np_array, sample_rate).
        """
        
        import pydub
        
        return (
            (numpy.array(
                audio.get_array_of_samples(),
                dtype = numpy.float64
            ).reshape(
                (audio.channels, -1)
            ) / (1 << (8 * audio.sample_width - 1))),
            audio.frame_rate,
        )
