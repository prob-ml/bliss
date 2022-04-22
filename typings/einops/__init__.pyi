from typing import Any, TypeVar, Union

from numpy import ndarray
from torch import Tensor

S = TypeVar('S', bound=Union[ndarray, Tensor])

def rearrange(tensor: S, pattern: str, **axes_lengths: int) -> S: ...

def reduce(tensor: S, pattern: str, reduction: Any, **axes_lengths: int) -> S: ...

def repeat(tensor: S, pattern: str, **axes_lengths: int) -> S: ...
