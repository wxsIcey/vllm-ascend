import torch
from torch import nn
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.config import VllmConfig, SpeculativeConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils

def get_eagle3_aux_layers_from_config(
    spec_config: SpeculativeConfig,
) -> tuple[int, ...] | None:
    if not (spec_config and spec_config.draft_model_config):
        return None
    hf_config = spec_config.draft_model_config.hf_config
    layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
    if not layer_ids:
        dflash_config = getattr(hf_config, "dflash_config", None)
        if dflash_config and isinstance(dflash_config, dict):
            # Add 1 to convert DFlash's aux layer id semantics
            layer_ids = [i + 1 for i in (dflash_config.get("target_layer_ids") or [])]
    if not layer_ids:
        dspark_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
        if dspark_layer_ids:
            layer_ids = [i + 1 for i in dspark_layer_ids]
    if not layer_ids:
        # Dense DSpark (e.g. Qwen3) also uses different aux layer semantics.
        target_layer_ids = getattr(hf_config, "target_layer_ids", None)
        if target_layer_ids:
            layer_ids = [i + 1 for i in target_layer_ids]
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(layer_ids)
    return None

eagle3_utils.get_eagle3_aux_layers_from_config = get_eagle3_aux_layers_from_config

class DSparkConfidenceHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        self.proj = ReplicatedLinear(
            config.hidden_size + rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden_states, markov_embeds], dim=-1) 
        confidence = self.proj(x.float()) 
        return confidence.squeeze(-1)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            rank,
            params_dtype=torch.float32,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.markov_w1(token_ids)
        logits = self.logits_processor(
            self.markov_w2,
            embeds.view(-1, embeds.shape[-1]).float(),
        )
        return logits.view(*embeds.shape[:-1], -1), embeds


ori_init = DFlashQwen3Model._init

def new_init(
    self,
    *,
    vllm_config: VllmConfig,
    start_layer_id: int = 0,
    prefix: str = "",
) -> None:
    hf_config = vllm_config.speculative_config.draft_model_config.hf_config 
    if hasattr(hf_config, "markov_head_type"):
        if not hasattr(hf_config, "dflash_config") or hf_config.dflash_config is None:
            hf_config.dflash_config = {}
            hf_config.dflash_config["target_layer_ids"] = hf_config.target_layer_ids
    ori_init(
        self,
        vllm_config=vllm_config,
        start_layer_id=start_layer_id,
        prefix=prefix,
    )
    if hasattr(hf_config, "markov_head_type"):
        self.markov_head = DSparkMarkovHead(vllm_config, prefix=f"{prefix}.markov_head")
        self.confidence_head = DSparkConfidenceHead(vllm_config, prefix=f"{prefix}.confidence_head")


DFlashQwen3Model._init = new_init