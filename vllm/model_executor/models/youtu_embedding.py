# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 Tencent Youtu Lab and the vLLM team.
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
"""Youtu-Embedding model for vLLM - Optimized for embedding inference."""

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, LoRAConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .interfaces import SupportsLoRA
from .interfaces_base import default_pooling_type
from .utils import (
    get_pp_missing_layer_names as get_pp_missing_layer_names_utils)
from .utils import is_pp_missing_parameter, make_layers


class UTURMSNorm(nn.Module):
    """RMSNorm for UTU model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class UTUMoEConfig:
    """Configuration for MoE.

    Not used in base Youtu-Embedding but kept for compatibility.
    """
    pass


class UTUMLP(nn.Module):
    """
    UTU MLP layer with SwiGLU activation.

    Architecture:
        gate_proj: hidden_size -> intermediate_size
        up_proj: hidden_size -> intermediate_size
        down_proj: intermediate_size -> hidden_size
        activation: SiLU(gate) * up
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Merge gate_proj and up_proj into single column parallel linear
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # [gate, up]
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        # SiLU activation for SwiGLU
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only 'silu' is supported for UTU models.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class UTUAttention(nn.Module):
    """
    UTU Attention with QK-Norm.

    Key features:
    - Multi-head attention with GQA support
    - Q and K normalization (QK-Norm)
    - RoPE position embeddings
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # Head dimension
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-Norm: normalize Q and K
        self.q_norm = UTURMSNorm(self.head_dim,
                                 eps=getattr(config, "rms_norm_eps", 1e-6))
        self.k_norm = UTURMSNorm(self.head_dim,
                                 eps=getattr(config, "rms_norm_eps", 1e-6))

        # RoPE
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # attn_cls = (
        #     EncoderOnlyAttention
        #     if attn_type == AttentionType.ENCODER_ONLY
        #     else Attention
        # )

        # Attention backend
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head attention
        # q: [batch * seq_len, num_heads, head_dim]
        # k, v: [batch * seq_len, num_kv_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # QK-Norm: Apply RMSNorm to Q and K
        # Reshape to apply norm along head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Flatten back
        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        if kv_cache is None:
            attn_output = self.attn(q, k, v, attn_metadata)
        else:
            attn_output = self.attn(q, k, v, attn_metadata, kv_cache)

        # Attention computation
        # attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class UTUDecoderLayer(nn.Module):
    """
    UTU Decoder Layer.

    Architecture:
        hidden_states -> input_layernorm -> self_attn -> residual
                      -> post_attention_layernorm -> mlp -> residual
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.hidden_size = config.hidden_size

        # RoPE configuration
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Attention
        self.self_attn = UTUAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )

        # MLP
        self.mlp = UTUMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )

        # Layer norms
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention with residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # MLP with residual
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class UTUModel(nn.Module):
    """
    UTU Model - Transformer decoder for embedding.

    This is the core transformer without any task-specific head.
    """
    is_pooling_model = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "utu_model",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: UTUDecoderLayer(
                vllm_config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        kv_caches: Optional[list[torch.Tensor]],
        attn_metadata: Optional[AttentionMetadata],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        if (intermediate_tensors is not None
                and get_pp_missing_layer_names_utils(self)
                and hasattr(intermediate_tensors, 'hidden_states')):
            return intermediate_tensors.hidden_states

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            kv_cache = kv_caches[i - self.start_layer] if (
                i - self.start_layer) < len(kv_caches) else None
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@default_pooling_type("MEAN")
class UTUForCausalLM(nn.Module, SupportsLoRA):
    is_pooling_model = True
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        vllm_config: VllmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = UTUModel(
            vllm_config,
            cache_config,
            quant_config,
            lora_config=lora_config,
            prefix=prefix,
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE),
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

        # Initialize pooler for embedding tasks
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config is not None:
            self.pooler = DispatchPooler({
                "encode":
                Pooler.for_encode(pooler_config),
                "embed":
                Pooler.for_embed(pooler_config),
            })
        else:
            # Default pooler if config is not provided
            from vllm.config import PoolerConfig
            default_pooler_config = PoolerConfig(pooling_type="MEAN")
            self.pooler = DispatchPooler({
                "encode":
                Pooler.for_encode(default_pooler_config),
                "embed":
                Pooler.for_embed(default_pooler_config),
            })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kv_caches is None and 'kv_cache' in kwargs:
            kv_caches = kwargs['kv_cache']
        if attn_metadata is None and 'attention_metadata' in kwargs:
            attn_metadata = kwargs['attention_metadata']

        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Remove "model." prefix if present (from HuggingFace checkpoint)
            if name.startswith("model."):
                name = name[6:]  # Remove "model." prefix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Handle Q/K norm weights
                if "q_norm" in name or "k_norm" in name:
                    full_name = f"model.{name}"
                    if full_name in params_dict:
                        param = params_dict[full_name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(full_name)
                        break
                full_name = f"model.{name}"
                if full_name in params_dict:
                    param = params_dict[full_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(full_name)
                    break
            else:
                # Handle Q/K norm weights
                if "q_norm" in name or "k_norm" in name:
                    full_name = f"model.{name}"
                    if name in params_dict:
                        param = params_dict[full_name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(full_name)
                        continue

                # Skip missing layers
                key_name = name
                full_name = f"model.{name}"
                if full_name in params_dict:
                    key_name = full_name
                elif name in params_dict:
                    key_name = name
                else:
                    #logger.warning(f"Parameter {name} not found in ")
                    break

                if is_pp_missing_parameter(key_name, self):
                    continue
                #print(f"===params_dict:{params_dict}, ===name:{name}")
                param = params_dict[key_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(key_name)


DEFAULT_VOCAB_PADDING_SIZE = 64
