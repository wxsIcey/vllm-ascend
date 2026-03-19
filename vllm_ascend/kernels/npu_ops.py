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
import torch
import torch_npu
from torch import Tensor
from vllm import ir


@ir.ops.rotary_embedding.register_impl("npu_kernels")
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor]:
    rotary_dim = cos_sin_cache.shape[-1]
    query_shape, key_shape = query.shape, key.shape
    if rotary_dim < head_size:
        num_tokens = query.shape[0]
        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]
        q_rot = q_rot.contiguous().view(num_tokens, -1)
        k_rot = k_rot.contiguous().view(num_tokens, -1)
        # only the rotary part is processed here,
        # the dimension should be rotary_dim
        torch_npu._npu_rotary_embedding(
            positions,
            q_rot,
            k_rot,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        )
        q_rot = q_rot.view(num_tokens, -1, rotary_dim)
        k_rot = k_rot.view(num_tokens, -1, rotary_dim)
        query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
    else:
        # TODO: Remove the contiguous in the future.
        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(key.shape[0], -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
    return query.view(query_shape), key.view(key_shape)
