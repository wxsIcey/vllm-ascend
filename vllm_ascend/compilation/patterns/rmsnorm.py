import torch
from .. import config

class RMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            """
            Generate example inputs for the AddRMSNormQuant fusion pattern.
            """
            rms_norm_input = torch.randn(2, 4, device="meta")
            residual = torch.randn(2, 4, device="meta")
            rms_norm_weight = torch.randn(4, device="meta")
            scale = torch.tensor([1.0], device="meta")
            offset = torch.tensor([0.0], device="meta")
            return [rms_norm_input, residual, rms_norm_weight, scale, offset]

        def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                    rms_norm_weight, 1e-6)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_quantize(
                out0, scale, offset, torch.qint8, -1, False)
            return quantized_output, out1


        def replacement(rms_norm_input, residual, rms_norm_weight, scale, offset):
            """
          Replacement for the AddRMSNormQuant fusion.
          """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                1. /
                scale,  # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
                offset,
                epsilon=1e-6)
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return (pattern, replacement, get_inputs())
    

def register_all_patterns():
    from . import register_pattern
    if config.compilation.fusion_patterns.enable_rms_norm_quant:
        pattern, replacement, example_inputs = RMSNormQuantPattern.create()
        register_pattern(
            "rms_norm_quant_pattern",
            pattern,
            replacement, 
            example_inputs,
        )
