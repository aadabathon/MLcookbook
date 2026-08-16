"""Serialization helpers shared by all result objects.

Every analysis in mlcookbook returns a frozen dataclass rather than printing
or plotting. ``to_jsonable`` converts such a dataclass (including nested
dataclasses, numpy scalars/arrays, and enums) into plain Python containers so
a future API layer can return ``result.to_dict()`` as JSON directly.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert *value* into JSON-serializable Python builtins."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None  # JSON has no NaN/Infinity
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


class Result:
    """Mixin giving result dataclasses a uniform ``to_dict()``.

    Kept as a plain mixin (not a dataclass base) so subclasses are free to
    order required and defaulted fields however they like.
    """

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
