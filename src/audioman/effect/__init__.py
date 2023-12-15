from typing import Literal

from .effect import Effect
from .adjustable_fade import AdjustableFade

effects: dict[Literal[
    'adjustable_fade',
    ], type[Effect]] = {
    'adjustable_fade': AdjustableFade,
}
