"""Full-stack segment definitions for the LFM2.5 surgical prefill pipeline.

Stage 0 deliberately isolated one block of each class. This module composes
those already-validated equations without changing them. Contiguous blocks of
the same class share one Core ML prediction boundary; the frozen checkpoint
therefore becomes thirteen layer-stack segments, not sixteen per-layer models.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .blocks import (
    ANELayoutMLP,
    CoreMLAttentionBlock,
    ReferenceAttentionBlock,
    ReferenceMLP,
    _copy_conv2d_weight,
    _copy_linear_weight,
    rms_norm_ane_explicit,
    rms_norm_reference,
)
from .checkpoint import Lfm2Config, load_block_tensors, load_tensors


@dataclass(frozen=True)
class SegmentSpec:
    """One maximal contiguous run of a single checkpoint block class."""

    index: int
    layer_type: str
    layer_indices: tuple[int, ...]

    @property
    def name(self) -> str:
        """Return the stable artifact stem for this segment."""

        first = self.layer_indices[0]
        last = self.layer_indices[-1]
        layer_range = str(first) if first == last else f"{first}_{last}"
        short_type = "gqa" if self.layer_type == "full_attention" else "conv"
        return f"segment_{self.index:02d}_{short_type}_{layer_range}"


def build_segment_specs(layer_types: tuple[str, ...]) -> tuple[SegmentSpec, ...]:
    """Partition ``layer_types`` into maximal same-class contiguous runs."""

    specs: list[SegmentSpec] = []
    cursor = 0
    for segment_index, (layer_type, group) in enumerate(groupby(layer_types)):
        run_length = sum(1 for _ in group)
        indices = tuple(range(cursor, cursor + run_length))
        specs.append(SegmentSpec(segment_index, layer_type, indices))
        cursor += run_length
    return tuple(specs)


class ReferenceConvBlockWithState(nn.Module):
    """Fp32 causal-convolution block with explicit incoming decode state."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize the reference equation from one real checkpoint layer."""

        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
        self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)
        self.mlp = ReferenceMLP(config, tensors)
        self.register_buffer("operator_norm_weight", tensors["operator_norm.weight"].float())
        self.register_buffer("ffn_norm_weight", tensors["ffn_norm.weight"].float())
        self.register_buffer("conv_weight", tensors["conv.conv.weight"].float())
        _copy_linear_weight(self.in_proj, tensors["conv.in_proj.weight"])
        _copy_linear_weight(self.out_proj, tensors["conv.out_proj.weight"])

    def forward(
        self, hidden_states: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one block and replace its explicit ``[B,H,k-1]`` live state."""

        residual = hidden_states.float()
        normalized = rms_norm_reference(
            residual, self.operator_norm_weight, self.config.norm_eps
        )
        projected = self.in_proj(normalized).transpose(1, 2)
        gate_b, gate_c, values = torch.chunk(projected, 3, dim=1)
        gated_values = gate_b * values
        history_and_prompt = torch.cat((conv_state.float(), gated_values), dim=-1)
        convolved = functional.conv1d(
            history_and_prompt,
            self.conv_weight,
            bias=None,
            groups=self.config.hidden_size,
        )
        hidden_states = residual + self.out_proj((gate_c * convolved).transpose(1, 2))
        ffn_input = rms_norm_reference(
            hidden_states, self.ffn_norm_weight, self.config.norm_eps
        )
        state_width = self.config.conv_l_cache - 1
        return (
            hidden_states + self.mlp(ffn_input),
            gated_values[..., -state_width:],
        )


class CoreMLConvBlockWithState(nn.Module):
    """Fp16 ANE-layout convolution block with explicit incoming live state."""

    def __init__(self, config: Lfm2Config, tensors: dict[str, torch.Tensor]) -> None:
        """Initialize the Stage 1 block from one real checkpoint layer."""

        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.in_proj = nn.Conv2d(
            hidden, 3 * hidden, 1, bias=config.conv_bias, dtype=torch.float16
        )
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, config.conv_l_cache),
            groups=hidden,
            bias=config.conv_bias,
            dtype=torch.float16,
        )
        self.out_proj = nn.Conv2d(
            hidden, hidden, 1, bias=config.conv_bias, dtype=torch.float16
        )
        self.mlp = ANELayoutMLP(config, tensors)
        self.register_buffer(
            "operator_norm_weight", tensors["operator_norm.weight"].to(torch.float16)
        )
        self.register_buffer(
            "ffn_norm_weight", tensors["ffn_norm.weight"].to(torch.float16)
        )
        _copy_conv2d_weight(self.in_proj, tensors["conv.in_proj.weight"])
        _copy_conv2d_weight(self.depthwise, tensors["conv.conv.weight"])
        _copy_conv2d_weight(self.out_proj, tensors["conv.out_proj.weight"])

    def forward(
        self, hidden_states: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one block on ``[B,H,1,S]`` and replace its explicit live state."""

        residual = hidden_states
        normalized = rms_norm_ane_explicit(
            residual, self.operator_norm_weight, self.config.norm_eps
        )
        gate_b, gate_c, values = torch.chunk(self.in_proj(normalized), 3, dim=1)
        gated_values = gate_b * values
        history_and_prompt = torch.cat(
            (conv_state.unsqueeze(2), gated_values), dim=-1
        )
        convolved = self.depthwise(history_and_prompt)
        hidden_states = residual + self.out_proj(gate_c * convolved)
        ffn_input = rms_norm_ane_explicit(
            hidden_states, self.ffn_norm_weight, self.config.norm_eps
        )
        state_width = self.config.conv_l_cache - 1
        return (
            hidden_states + self.mlp(ffn_input),
            gated_values[:, :, 0, -state_width:],
        )


class CoreMLLayerSegment(nn.Module):
    """One class-contiguous fp16 layer-stack segment with flat tensor I/O."""

    def __init__(
        self, config: Lfm2Config, checkpoint_dir, spec: SegmentSpec
    ) -> None:
        """Load exactly the real checkpoint layers named by ``spec``."""

        super().__init__()
        self.spec = spec
        blocks: list[nn.Module] = []
        for layer_index in spec.layer_indices:
            tensors = load_block_tensors(checkpoint_dir, layer_index, spec.layer_type)
            if spec.layer_type == "conv":
                blocks.append(CoreMLConvBlockWithState(config, tensors))
            elif spec.layer_type == "full_attention":
                blocks.append(CoreMLAttentionBlock(config, tensors))
            else:
                raise ValueError(f"unsupported segment type: {spec.layer_type}")
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, hidden_states: torch.Tensor, *segment_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run the segment and emit hidden output plus per-layer state outputs."""

        outputs: list[torch.Tensor] = []
        if self.spec.layer_type == "conv":
            if len(segment_inputs) != len(self.blocks):
                raise ValueError("one explicit state input is required per conv layer")
            for block, state in zip(self.blocks, segment_inputs, strict=True):
                hidden_states, state_out = block(hidden_states, state)
                outputs.append(state_out)
        else:
            if len(segment_inputs) != 3:
                raise ValueError("GQA segments require cosine, sine, and attention mask")
            cosine, sine, attention_mask = segment_inputs
            for block in self.blocks:
                hidden_states, key_out, value_out = block(
                    hidden_states, cosine, sine, attention_mask
                )
                outputs.extend((key_out, value_out))
        return (hidden_states, *outputs)


class ReferenceLayerStack(nn.Module):
    """Independent fp32 full layer stack used as the Stage 1 numerical oracle."""

    def __init__(self, config: Lfm2Config, checkpoint_dir) -> None:
        """Load all sixteen real layers in their frozen checkpoint order."""

        super().__init__()
        blocks: list[nn.Module] = []
        for layer_index, layer_type in enumerate(config.layer_types):
            tensors = load_block_tensors(checkpoint_dir, layer_index, layer_type)
            if layer_type == "conv":
                blocks.append(ReferenceConvBlockWithState(config, tensors))
            else:
                blocks.append(ReferenceAttentionBlock(config, tensors))
        self.blocks = nn.ModuleList(blocks)
        self.layer_types = config.layer_types

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        attention_mask: torch.Tensor,
        *conv_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run all layers and emit hidden output, conv states, then GQA K/V pairs."""

        conv_cursor = 0
        conv_outputs: list[torch.Tensor] = []
        cache_outputs: list[torch.Tensor] = []
        for layer_type, block in zip(self.layer_types, self.blocks, strict=True):
            if layer_type == "conv":
                hidden_states, state_out = block(
                    hidden_states, conv_states[conv_cursor]
                )
                conv_cursor += 1
                conv_outputs.append(state_out)
            else:
                hidden_states, key_out, value_out = block(
                    hidden_states, cosine, sine, attention_mask
                )
                cache_outputs.extend((key_out, value_out))
        return (hidden_states, *conv_outputs, *cache_outputs)


class CoreMLLayerStack(nn.Module):
    """Fp16 monolithic layer-stack control with the same state contract."""

    def __init__(self, config: Lfm2Config, checkpoint_dir) -> None:
        """Load all sixteen real layers without introducing segment boundaries."""

        super().__init__()
        blocks: list[nn.Module] = []
        for layer_index, layer_type in enumerate(config.layer_types):
            tensors = load_block_tensors(checkpoint_dir, layer_index, layer_type)
            if layer_type == "conv":
                blocks.append(CoreMLConvBlockWithState(config, tensors))
            else:
                blocks.append(CoreMLAttentionBlock(config, tensors))
        self.blocks = nn.ModuleList(blocks)
        self.layer_types = config.layer_types

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        attention_mask: torch.Tensor,
        *conv_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run all layers and emit hidden output, conv states, then GQA K/V pairs."""

        conv_cursor = 0
        conv_outputs: list[torch.Tensor] = []
        cache_outputs: list[torch.Tensor] = []
        for layer_type, block in zip(self.layer_types, self.blocks, strict=True):
            if layer_type == "conv":
                hidden_states, state_out = block(
                    hidden_states, conv_states[conv_cursor]
                )
                conv_cursor += 1
                conv_outputs.append(state_out)
            else:
                hidden_states, key_out, value_out = block(
                    hidden_states, cosine, sine, attention_mask
                )
                cache_outputs.extend((key_out, value_out))
        return (hidden_states, *conv_outputs, *cache_outputs)


class CoreMLEmbedding(nn.Module):
    """Token embedding model with an explicit mask for left padding."""

    def __init__(self, checkpoint_dir) -> None:
        """Load the frozen tied embedding matrix as fp16 weights."""

        super().__init__()
        name = "model.embed_tokens.weight"
        weight = load_tensors(checkpoint_dir, {name})[name].to(torch.float16)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)

    def forward(
        self, token_ids: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return masked ANE-layout hidden states shaped ``[1,H,1,S]``."""

        hidden = self.embedding(token_ids).permute(0, 2, 1).unsqueeze(2)
        return hidden * token_mask


class CoreMLLMHead(nn.Module):
    """Final RMSNorm plus tied language-model projection for one position."""

    def __init__(self, config: Lfm2Config, checkpoint_dir) -> None:
        """Load the final norm and tied embedding matrix from the checkpoint."""

        super().__init__()
        names = {"model.embedding_norm.weight", "model.embed_tokens.weight"}
        tensors = load_tensors(checkpoint_dir, names)
        self.epsilon = config.norm_eps
        self.register_buffer(
            "norm_weight", tensors["model.embedding_norm.weight"].to(torch.float16)
        )
        self.projection = nn.Conv2d(
            config.hidden_size,
            config.vocab_size,
            1,
            bias=False,
            dtype=torch.float16,
        )
        _copy_conv2d_weight(self.projection, tensors["model.embed_tokens.weight"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for one orchestrator-selected position."""

        normalized = rms_norm_ane_explicit(
            hidden_states, self.norm_weight, self.epsilon
        )
        return self.projection(normalized)[:, :, 0, 0]
