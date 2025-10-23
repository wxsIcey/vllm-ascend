from functools import lru_cache
from typing import Any, Callable, List

import torch

from ..passes.pattern_match_pass import PatternMatchPass

def register_pattern(
    name: str,
    pattern: Callable[..., Any],
    replacement: Callable[..., Any],
    example_inputs: List[torch.Tensor],
):
    
    PatternMatchPass().register_pattern(name, pattern, replacement, example_inputs)

@lru_cache(None)
def lazy_init():
    from ..patterns import rms_norm
    rms_norm.register_all_patterns()