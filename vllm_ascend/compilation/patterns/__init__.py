from functools import lru_cache
from typing import Any, Callable, List

import torch

from ..passes.pattern_match_pass import PatternMatchPass
from . import rmsnorm
    
def register_pattern(
    name: str,
    pattern: Callable[..., Any],
    replacement: Callable[..., Any],
    example_inputs: List[torch.Tensor],
):
    PatternMatchPass().register_pattern(name, pattern, replacement, example_inputs)

@lru_cache(None)
def lazy_init():
    rmsnorm.register_all_patterns()