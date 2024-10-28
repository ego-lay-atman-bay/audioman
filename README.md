# audioman
Lightweight Python library for manipulating audio files. This can be used for editing audio, or tagging audio.

## Installation

Install `audioman` using pip

```shell
pip install audioman
```

## Audio

Audio editing here is done using the audio samples represented in a numpy array. Although, it is easy to combine audio together without touching the samples.

First we load the audio.

```python
from audioman import Audio

audio = Audio('audio.wav')
```

Or we can load using raw samples

```python
import numpy
from audioman import Audio

audio = Audio(numpy.array([[0,0,0,0,0]]), sample_rate = 4400)
```

Once you're done modifying the audio file, you can save it.

```python
audio.save('modified.wav')
```

If you don't specify a filename, it will use `self.filename`, but since that is usually lost when modifying the audio, it's best to specify the filename anyway.

`Audio` objects are can be combined

```python
from audioman import Audio

audio1 = Audio('audio1.wav')
audio2 = Audio('audio2.wav')

audio3 = audio1 + audio2

len(audio3) # len(audio1) + len(audio2)
```

You can also increase the gain.

```python
audio = Audio('audio.wav')
audio2 = audio + 5
```

The original audio object does not get modified.

Trimming audio is also easy.

```python
audio = Audio('audio.wav')
trimmed = audio.trim(start = 0, length = audio.seconds_to_samples(3))
```

Of course if you only specify the length, the start will be assumed to be 0.

```python
trimmed = audio.trim(audio.seconds_to_samples(3))
trimmed.samples_to_seconds(len(trimmed)) # 3.0
```

If you wish to, you can also split the audio in 2 audio tracks.

```python
audio1, audio2 = audio.split()
```

By default it will split in half, but you can specify the middle.

```python
audio1, audio2 = audio.split(audio.seconds_to_samples(3))
```

If you wish to do so, you can add silence to an audio track.

```python
audio2 = audio.add_silence(start = 0, length = audio.seconds_to_samples(3))
```

Of course if you want to add to the end, you can use `-1`

```python
audio2 = audio.add_silence(start = -1, length = audio.seconds_to_samples(3))
```

If you omit `start`, it will set it to 0.

```python
audio2 = audio.add_silence(audio.seconds_to_samples(3))
```

If you need to mix (overlay) two audio tracks together, use the `.mix()` method.

```python
audio = Audio('audio.wav')
drums = Audio('drums.wav')

mixed = audio.mix(drums)
```

## Effects

There are effects that you can apply to audio.

To use effects, create an effect object, and apply the options.

```python
from audioman import Audio
from audioman.effect import AdjustableFade

audio = Audio('audio.wav')

fade = AdjustableFade(length = audio.seconds_to_samples(3), gain0 = 0, gain1 = 1, fade_adjust = -1)
faded = audio.apply_effect(fade, start = audio.seconds_to_samples(-3))
```

You can find all the options for an effect in the `.OPTIONS` property.

```python
from audioman.effect import AdjustableFade

print(AdjustableFade.OPTIONS)

""" stdout:
{
    'gain0': {
        'default': 1,
        'type': float,
    },
    'gain1': {
        'default': 0,
        'type': float,
    },
    'curve_ratio': {
        'default': 0,
        'type': float,
    }
}
"""
```

## PyDub support

While pydub is in fact a dependency, and there is a function to be able to convert to modify an `Audio` object with pydub, it is not really used much in the library. Some effects used pydub to make it easier to code, but I have since recreated them with my library.
