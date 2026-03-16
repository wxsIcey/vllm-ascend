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


@ir.ops.rms_norm_gated.register_impl("npu_kernels", supported=True)
def rms_norm_gated(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    z: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "",
) -> Tensor:
    from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    y, _, _ = layer_norm_fwd_npu(
        x,
        weight,
        bias,
        epsilon,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )
    return y.reshape(x_shape_og)
