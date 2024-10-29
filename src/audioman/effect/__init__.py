from typing import Literal

from .effect import Effect
from .adjustable_fade import AdjustableFade
from .amplify import Amplify
from .truncate_silence import TruncateSilence
from .normalize import Normalize
from .set_volume import SetVolume
from .clamp import Clamp

EFFECTS: dict[Literal[
    'adjustable_fade',
    'amplify',
    'truncate_silence',
    'remove_silence',
    'normalize',
    'set_volume',
    'clamp',
], type[Effect]] = {
    'adjustable_fade': AdjustableFade,
    'amplify': Amplify,
    'truncate_silence': TruncateSilence,
    'remove_silence': TruncateSilence,
    'normalize': Normalize,
    'set_volume': SetVolume,
    'clamp': Clamp,
}
