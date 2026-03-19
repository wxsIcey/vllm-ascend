#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from torch import Tensor
from vllm import ir
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm_ascend.ops.triton.rope import rope_forward_triton


@ir.ops.rotary_embedding.register_impl("npu_triton", supported=HAS_TRITON)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor]:
    query_shape, key_shape = query.shape, key.shape
    rotary_dim = cos_sin_cache.shape[-1]
    num_tokens = query.shape[0]
    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)
    query, key = rope_forward_triton(
        query.view(num_tokens, -1, head_size),
        key.view(num_tokens, -1, head_size),
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        rope_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )
    return query.view(query_shape), key.view(key_shape)
