import logging
import math
from typing import TYPE_CHECKING

import numpy

from .effect import Effect

if TYPE_CHECKING:
    from ..audio import Audio

class AdjustableFade(Effect):
    """An adjustable fade. This is basically taken directly from audacity, with only the linear fade.

    Options:
        gain0 (float): Start gain. Value between 0 and 1. Defaults to 1.
        gain1 (float): End gain. Value between 0 and 1. Defaults to 0.
        curve_ratio (float): Curve adjust ratio. Value between -1 and 1. Defaults to 0.
    """
    OPTIONS = {
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
        },
        'type': {
            'default': 'linear',
            'type': str,
            'options': [
                'linear',
                's-curve',
            ]
        }
    }
    
    TYPE = 'audio'
    
    def __init__(
        self,
        options: dict = {},
        **kwargs,
    ) -> None:
        """Audio effect. This effect will last the specified length in samples. If the duration is specified (in seconds), then it will use that length. This will assume the sample rate is `4400`, unless specified.

        Args:
            **options (dict[str, Any]): Effect options.
        
        Options:
            gain0 (float): Start gain. Value between 0 and 1. Defaults to 1.
            gain1 (float): End gain. Value between 0 and 1. Defaults to 0.
            curve_ratio (float): Curve adjust ratio. Value between -1 and 1. Defaults to 0.
        """
        super().__init__(
            options,
            **kwargs,
        )
    
    def get(self) -> numpy.ndarray:
        return 
    
    def apply(self, audio: 'Audio') -> 'Audio':
        if self.options['type'] == 'linear':
            return audio.copy(audio.samples * self.linear_fade(
                gain0 = self.options['gain0'],
                gain1 = self.options['gain1'],
                curve_ratio = self.options['curve_ratio'],
                length = len(audio),
            ))
        elif self.options['type'] == 's-curve':
            return self.raised_cos(
                gain0 = self.options['gain0'],
                gain1 = self.options['gain1'],
                curve_ratio = self.options['curve_ratio'],
                audio = audio,
            )

    def linear_fade(
        self,
        gain0: float,
        gain1: float,
        curve_ratio: float,
        length: int,
    ) -> numpy.ndarray[float]:
        """Generate fade scaler for the specified length.

        Args:
            gain0 (float): Start gain
            gain1 (float): End gain
            curve_ratio (float): Mid-fade adjust (%) between 0 and 1
            length (int): Sample length of the fade (not sample rate).

        Returns:
            numpy.ndarray[float]: numpy array
        """
        if (gain0 == gain1):
            return numpy.array([gain0] * length)
        elif ((curve_ratio > 0) and (curve_ratio < 0.5)):
            _curve_ratio = curve_ratio * 2
            return (
                (
                    self.scale_curve(
                        gain0,
                        gain1,
                        self.linear(
                            gain0,
                            gain1,
                            length,
                        )
                    ) * (1 - _curve_ratio)
                ) + (
                    self.scale_curve(
                        gain0,
                        gain1,
                        self.cosine_curve(gain0, gain1, length),
                    ) * _curve_ratio
                )
            )
        elif (curve_ratio > 0):
            return self.cos_curve(gain0, gain1, 1.5 - curve_ratio, length)
        else:
            return self.simple_curve(gain0, gain1, (1 - (2 * curve_ratio)), length)
    

    def simple_curve(
        self,
        gain0: float,
        gain1: float,
        power: float,
        length: int,
    ) -> numpy.ndarray[float]:
        return self.curve_adjust(gain0, gain1, power, self.linear(gain0, gain1, length))
        
    def scale_curve(
        self,
        gain0: float,
        gain1: float,
        env: numpy.ndarray[float],
    ) -> numpy.ndarray[float]:
        return min(gain0, gain1) + (abs(gain0 - gain1) * env)

    def linear(
        self,
        gain0: float,
        gain1: float,
        length: int,
    ) -> numpy.ndarray[float]:
        return numpy.linspace(gain0, gain1, length)
    
    # s-curve Raised cosine fades.
    def raised_cos(
        self,
        gain0: float,
        gain1: float,
        curve_ratio: float,
        length: 'Audio',
    ):
        import pwlf
        
        _curve_ratio = self.exp_scale_mid(2 * curve_ratio) if curve_ratio > 0 else self.exp_scale_mid(1.63 * curve_ratio)

        if gain0 == gain1:
            return numpy.array([gain0] * length)
        elif gain0 > gain1:
            return numpy.exp(numpy.multiply(pwlf.PiecewiseLinFit([0, 1], [1 - _curve_ratio])))

    def cosine_curve(
        self,
        gain0: float,
        gain1: float,
        length: int,
    ) -> numpy.ndarray[float]:
        """Creates half a cosine wave of the specified length

        Args:
            gain0 (float): gain0
            gain1 (float): gain1
            length (int): length

        Returns:
            numpy.ndarray: numpy array
        """
        phase = 1 if (gain0 > gain1) else -1
        return (numpy.cos(numpy.deg2rad((numpy.arange(length) / length) * 180)) * phase + 1) * 0.5

    def cos_curve(
        self,
        gain0: float,
        gain1: float,
        power: float,
        length: int,
    ) -> numpy.ndarray[float]:
        return self.curve_adjust(gain0, gain1, power, self.cosine_curve(gain0, gain1, length))

    def curve_adjust(
        self,
        gain0: float,
        gain1: float,
        power: float,
        env: numpy.ndarray[float],
    ) -> numpy.ndarray[float]:
        return self.scale_curve(gain0, gain1,
            (env if (power == 1) else numpy.exp(power * numpy.log(env))))


    def raised_cosine(
        self,
        phase,
    ):
        pass
    
    def log_exp_curve(
        self,
        x,
        direction,
    ):
        pass

    def exp_scale_mid(
        self,
        x,
    ):
        return (numpy.exp(1 - x) - numpy.e) / (1 - numpy.e)

    def gainscale(
        self,
        gain,
    ):
        pass
