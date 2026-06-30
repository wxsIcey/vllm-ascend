# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)

class DSparkSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )
        self.context_positions = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        self.dflash_causal = False
        
        # 这里要打个patch get_parallel_drafting_token_id
        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )
        
    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config

        from vllm.compilation.backends import set_model_tag

        # DSpark uses non-causal attention.
        causal = False
        draft_vllm_config = replace(
            self.vllm_config,
            attention_config=replace(
                self.vllm_config.attention_config,
                use_non_causal=not causal,
                backend=speculative_config.attention_backend,
            ),
        )

        with set_model_tag("dspark_head"):
            draft_model = get_model(
                vllm_config=draft_vllm_config, model_config=draft_model_config
            )

        if get_pp_group().world_size != 1:
            raise NotImplementedError("DSpark does not support pipeline parallelism.")

        # Self-contained dense DSpark drafts (e.g. Qwen3) ship their own embed_tokens
        # and lm_head, so aliasing the target's would clobber the loaded weights.
        # Only the DeepSeek-V4 draft (weights in the target's checkpoint) shares.
        if not getattr(draft_model, "dspark_shares_target_embeddings", True):
            return draft_model

        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = target_language_model.model
        draft_inner = draft_model.model

        # Share the vocab embedding (target.model.embed_tokens -> draft.model).
        target_embed = getattr(target_inner, "embed_tokens", None)
        if target_embed is not None:
            if getattr(draft_inner, "embed_tokens", None) is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

        # Share the LM head (target.lm_head -> draft.lm_head).
        target_lm_head = getattr(target_model, "lm_head", None)
        if target_lm_head is not None:
            if getattr(draft_model, "lm_head", None) is not None:
                del draft_model.lm_head
            draft_model.lm_head = target_lm_head

        return draft_model
    
    def set_attn(self, model_state, kv_cache_config, block_tables) -> None:
        
        # 这里的代码需要再看一下
        super().set_attn(model_state, kv_cache_config, block_tables)
        
        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "DSpark found no draft attention groups."

        # Kept for compatibility with the parent class: DSpark generalizes to support
        # multiple KV cache groups.
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
        self.draft_block_size = block_tables.block_sizes[self.draft_kv_cache_group_id]

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to.
        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
        }
        gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}

        self._layer_group_idx = [
            gid_to_idx[name_to_gid[name]]
            for name in self.model.get_draft_kv_cache_layer_names()
        ]
        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        
    @torch.inference_mode()
    def propose( 
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )
        
        assert aux_hidden_states is not None, (
            "DSpark requires auxiliary target hidden states (dspark_target_layer_ids)."
        )
        hidden_states = self.model.combine_hidden_states(
            torch.cat(aux_hidden_states, dim=-1)
        )
        
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
        
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        
        if dummy_run and skip_attn_for_dummy_run:
            # Memory-profiling path: block tables / kv-cache not initialized.
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]
        
        
        # Support multiple KV cache groups
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            # 该triton算子的输入
            prepare_dspark_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions, 
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.block_tables.input_block_tables[gid],
                self.block_tables.block_sizes[gid],
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
            )
            
        
        
            

@triton.jit
def _prepare_dspark_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    # Anchor at offset 0 (the bonus token); noise tokens at offsets 1..N-1.
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_anchor = is_query & (query_off == 0)
    input_id = tl.where(is_anchor, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # DSpark samples at EVERY query position (the anchor predicts the first draft
    # token). Each position k predicts the NEXT token, so the sampled token's
    # position is query_pos + 1.
    is_sample = is_query
    sample_idx = req_idx * num_speculative_steps + query_off
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos + 1, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens: absolute length the draft attention reads up to (context +
        # the N query tokens).
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dspark_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dspark_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
            

        
        
        
        
        
        
        
        
        
    
        
        
        
        