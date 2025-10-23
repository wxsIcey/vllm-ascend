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
    
    
    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        """
        Process the FX graph and perform custom operation fusing.

        Args:
            graph (fx.Graph): The FX graph to be processed.
            example_inputs (optional): Example inputs for the graph.

        Returns:
            fx.Graph: The processed FX graph with custom operation fusing applied.
        """
        graph = self.compile(graph, example_inputs)
        return graph

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs,
        **kwargs,
    ) -> tuple[Callable, Optional[Any]]:
        def compile_inner(fx_graph, inputs):
            self.apply_pattern_match_passes(fx_graph)
            self.apply_decompose_auto_functionalized_pass(fx_graph)
            return fx_graph

        # Use the default decomposition table to decompose operators.
        decompositions = select_decomp_table()
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(fw_compiler=compile_inner, decompositions=decompositions)(
            gm, example_inputs
        )

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
        
    def compute_hash(self, config: Any) -> str:
        """
        Compute a unique hash for the compiler backend based on its configuration.

        Args:
            config: The configuration object for the compiler.

        Returns:
            A string representing the unique hash of the compiler backend.
        """
        # 示例：根据配置生成哈希值
        import hashlib
        config_str = str(config)  # 将配置对象转换为字符串
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()


