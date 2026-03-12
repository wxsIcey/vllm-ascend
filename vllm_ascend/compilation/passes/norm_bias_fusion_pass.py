#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import enable_custom_op


class AddRMSNormBiasPattern(BasePattern):
    """
    Replace npu_add_rms_norm (standard NPU op) with npu_add_rms_norm_bias
    (custom fused op) when the custom op library is available.
    This variant handles the case with no norm bias.
    """

    def get_inputs(self):
        x = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [x, residual, weight]

    def get_pattern(self):
        def pattern(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
            output = torch.ops.npu.npu_add_rms_norm(x, residual, weight, self.eps)
            return output[0], output[2]

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, self.eps)
            return output[0], output[2]

        return replacement


class AddRMSNormBiasPatternWithBias(BasePattern):
    """
    Replace npu_add_rms_norm + add_bias with npu_add_rms_norm_bias
    when the custom op library is available.
    This variant handles the case with a non-zero norm bias.
    """

    def get_inputs(self):
        x = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        weight = torch.randn(4, device="npu", dtype=self.dtype)
        bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [x, residual, weight, bias]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            output = torch.ops.npu.npu_add_rms_norm(x, residual, weight, self.eps)
            normed_x = output[0] + bias
            return normed_x, output[2]

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, bias, self.eps)
            return output[0], output[2]

        return replacement


class AddRMSNormBiasPass(VllmInductorPass):
    """
    Compilation pass that upgrades npu_add_rms_norm to npu_add_rms_norm_bias
    when the custom op library is available. This allows downstream fusion
    passes (e.g. AddRMSNormQuantFusionPass) to match on npu_add_rms_norm_bias.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="norm_bias_fusion_pass")

        if not enable_custom_op():
            return

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("AddRMSNormBiasPass not enabled: unsupported dtype %s", dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            AddRMSNormBiasPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            AddRMSNormBiasPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
