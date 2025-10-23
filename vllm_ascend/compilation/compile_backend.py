import functools
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from vllm.compilation.compiler_interface import CompilerInterface
from . import patterns
from .passes.pattern_match_pass import PatternMatchPass
from . import config

# model = MyModel()
# compiled_model = torch.compile(model, backend=CompilerBackend()) 想一下在 vllm 中如何调用

class CompilerBackend(CompilerInterface):
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing etc.
    """
    name = "AscendCompilerBackend"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Callable, Optional[Any]]:
        def compile_inner(fx_graph, inputs):
            self.apply_pattern_match_passes(fx_graph)
            self.apply_decompose_auto_functionalized_pass(fx_graph)
            for node in fx_graph.graph.nodes:
                if node.op == "output":
                    output_types = tuple(
                        type(arg) for arg in node.args[0]
                    ) if isinstance(node.args[0], (list, tuple)) else (type(node.args[0]),)
                    if len(output_types) == 1 and output_types[0] is type(None):
                        raise RuntimeError(
                            "Graph output must be a (). This is so that we can avoid pytree processing of the outputs. "
                            "Please change the module to have tuple outputs or use aot_module instead."
                        )
            
            return fx_graph
        # Use the default decomposition table to decompose operators.
        decompositions = select_decomp_table()
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(fw_compiler=compile_inner, decompositions=decompositions)(
            graph, example_inputs
        ), None

    def apply_pattern_match_passes(self, graph: fx.GraphModule):
        patterns.lazy_init()
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="pattern_match_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(graph, f"pattern_match_pass").apply_gm_pass(
            PatternMatchPass()
        )

    def apply_decompose_auto_functionalized_pass(self, graph: fx.GraphModule):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="decompose_auto_functionalized_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(graph, "decompose_auto_functionalized").apply_graph_pass(
            decompose_auto_functionalized
        )

