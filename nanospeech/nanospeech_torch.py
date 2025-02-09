"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from functools import partial
import math
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast

import torchaudio

from einops import rearrange, reduce, repeat
import einx

from vocos import Vocos

from nanospeech.utils import fetch_from_hub

import safetensors.torch

# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(
    t: int["b"],  # noqa: F821
    length: int | None = None,
) -> bool["b n"]:  # noqa: F722
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(
    start: int["b"],  # noqa: F722 F821
    end: int["b"],  # noqa: F722 F821
    max_seq_len: int,
):
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(
    seq_len: int["b"],  # noqa: F722 F821
    frac_lengths: float["b"],  # noqa: F722 F821
    max_seq_len: int,
):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(start, end, max_seq_len)


# vocab-based tokenizer


def list_str_to_vocab_tensor(text: list[str], vocab: dict[str, int], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [[vocab.get(c, 0) for c in t] for t in text]
    list_tensors = [torch.LongTensor(t) for t in list_tensors]
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# utf-8 tokenizer


def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# raw wav to mel spec


class MelSpec(nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        normalize=False,
        power=1,
        norm=None,
        center=True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sample_rate = target_sample_rate

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=power,
            center=center,
            normalized=normalize,
            norm=norm,
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        mel = self.mel_stft(inp)
        mel = mel.clamp(min=1e-5).log()
        return mel


# RoPE


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None, persistent=False)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale, persistent=False)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(self, t):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")

        freqs = torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if not exists(self.scale):
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.stack((scale, scale), dim=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)


# global response normalization layer


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 block


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2

        # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise conv
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = rearrange(x, "b n d -> b d n")
        x = self.dwconv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


# feed forward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        query = rearrange(query, "b n (h d) -> b h n d", h=self.heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.heads)

        # mask. e.g., inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = rearrange(mask, "b n -> b 1 1 n")
            attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        x = rearrange(x, "b h n d -> b n (h d)")
        x = x.to(query.dtype)

        x = self.to_out(x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# text embedding


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = start.unsqueeze(1) + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text).long()

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
        drop_audio_cond=False,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# AdaLNZero


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# diffusion transformer


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim, bias=False)
        nn.init.zeros_(self.proj_out.weight)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, " -> b", b=batch)

        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output


# duration predictor

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = einx.where("b n, b n d, -> b n d", mask, t, 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.float(), "b n -> b", "sum")

    return einx.divide("b d, b -> b d", num, den.clamp(min=1.0))


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], text_embed: float["b n d"]):  # noqa: F722
        x = self.proj(torch.cat((x, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DurationBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, mask=None, rope=None):
        norm = self.attn_norm(x)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + attn_output
        norm = self.ff_norm(x)
        ff_output = self.ff(norm)
        x = x + ff_output
        return x


class DurationTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim

        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DurationBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.RMSNorm(dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        seq_len = x.shape[1]

        text_embed = self.text_embed(text, seq_len)
        x = self.input_embed(x, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope=rope)

        x = self.norm_out(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        mel_spec_kwargs: dict = dict(),
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        super().__init__()

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self.mel_spec.n_mel_channels

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.to_pred = nn.Sequential(nn.Linear(transformer.dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ..."))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        return_loss=False,
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, device = *inp.shape[:2], self.device

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                text = self.tokenizer(text).to(device)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = inp.new_zeros(batch).uniform_(0, 1)
            rand_index = (rand_frac_index * lens).long()

            seq = torch.arange(seq_len, device=device)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending

        inp = torch.where(
            repeat(mask, "b n -> b n d", d=self.num_channels),
            inp,
            torch.zeros_like(inp),
        )

        x = self.transformer(inp, text=text)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        if not return_loss:
            return pred

        # loss

        duration = lens.float() / SAMPLES_PER_SECOND
        return F.l1_loss(pred, duration)


# ode solvers


def odeint_euler(func, y0, t, **kwargs):
    """
    Solves ODE using the Euler method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # compute the next value
        k = func(t_current, y_current)
        y_next = y_current + dt * k
        ys.append(y_next)
        y_current = y_next

    return torch.stack(ys)


def odeint_midpoint(func, y0, t, **kwargs):
    """
    Solves ODE using the midpoint method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # midpoint approximation
        k1 = func(t_current, y_current)
        mid = y_current + 0.5 * dt * k1

        # compute the next value
        k2 = func(t_current + 0.5 * dt, mid)
        y_next = y_current + dt * k2
        ys.append(y_next)
        y_current = y_next

    return torch.stack(ys)


def odeint_rk4(func, y0, t, **kwargs):
    """
    Solves ODE using the Runge-Kutta 4th-order (RK4) method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # rk4 steps
        k1 = func(t_current, y_current)
        k2 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k1)
        k3 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k2)
        k4 = func(t_current + dt, y_current + dt * k3)

        # compute the next value
        y_next = y_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y_next)
        y_current = y_next

    return torch.stack(ys)


# conditional flow matching


class Nanospeech(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        duration_predictor: nn.Module | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        vocoder: Callable[[float["b d n"]]] | None = None,  # noqa: F722
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self.mel_spec.n_mel_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # tokenizer
        self.tokenizer = tokenizer

        # vocoder
        self.vocoder = vocoder

        # duration predictor
        self._duration_predictor = duration_predictor

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = (*inp.shape[:2], inp.dtype, self.device)

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                text = self.tokenizer(text).to(device)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally

        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, seq_len)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # timestep

        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample x(t)

        t = rearrange(time, "b -> b 1 1")
        w = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling

        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate

        rand_audio_drop = torch.rand(1, device=device)
        rand_cond_drop = torch.rand(1, device=device)
        drop_audio_cond = rand_audio_drop < self.audio_drop_prob
        drop_text = rand_cond_drop < self.cond_drop_prob
        drop_audio_cond = drop_audio_cond | drop_text

        pred = self.transformer(
            x=w,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
        )

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask].mean()

        return loss

    def predict_duration(
        self,
        cond: float["b n d"],  # noqa: F722
        text: int["b nt"],  # noqa: F722
        speed: float = 1.0,
    ) -> int:
        duration_in_sec = self._duration_predictor(cond, text)
        frame_rate = self.mel_spec.sample_rate // HOP_LENGTH
        duration = (duration_in_sec * frame_rate / speed).long()
        return duration

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"] | None = None,  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        ode_method: Literal["euler", "midpoint", "rk4"] = "rk4",
        steps=32,
        cfg_strength=1.0,
        speed=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
    ):
        self.eval()
        
        cond = torch.Tensor(cond).to(self.device)

        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()

        # handle raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond).to(self.device)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                text = self.tokenizer(text).to(device)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert text.shape[0] == batch

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)

        if cond_seq_len < text.shape[1]:
            cond_seq_len = text.shape[1]
            cond = F.pad(cond, (0, 0, 0, cond_seq_len - cond.shape[1]), value=0.0)

        # duration

        if duration is None and self._duration_predictor is not None:
            duration = self.predict_duration(cond, text, speed)
        elif duration is None:
            raise ValueError("Duration must be provided or a duration predictor must be set.")

        cond_mask = lens_to_mask(lens)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        assert lens < duration, "duration must be at least as long as the input"

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = rearrange(cond_mask, "... -> ... 1")

        # at each step, conditioning is fixed

        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            # save memory and speed up, as single inference needs no mask
            mask = None

        # neural ode

        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
            )

            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )

            output = pred + (pred - null_pred) * cfg_strength
            return output

        if ode_method == "euler":
            odeint_fn = odeint_euler
        elif ode_method == "rk4":
            odeint_fn = odeint_rk4
        elif ode_method == "rk4":
            odeint_fn = odeint_midpoint
        else:
            raise ValueError(f"Unknown method: {ode_method}")

        # noise input

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t = torch.linspace(0, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint_fn(fn, y0, t)

        sampled = trajectory[-1]
        
        # trim the reference audio
        
        out = sampled[:, cond_seq_len:]

        # vocode to a waveform
        
        if exists(self.vocoder):
            out = out.permute(0, 2, 1)
            out = self.vocoder(out.cpu())

        return out, trajectory

    @classmethod
    def from_pretrained(cls, hf_model_name_or_path: str) -> Nanospeech:
        path = fetch_from_hub(hf_model_name_or_path)

        if path is None:
            raise ValueError(f"Could not find model {hf_model_name_or_path}")

        # tokenizer

        vocab_path = path / "vocab.txt"
        vocab = {v: i for i, v in enumerate(Path(vocab_path).read_text().split("\n"))}
        if len(vocab) == 0:
            raise ValueError(f"Could not load vocab from {vocab_path}")
        tokenizer = partial(list_str_to_vocab_tensor, vocab=vocab)

        # vocoder

        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        # duration predictor

        duration_model_filename = "duration.safetensors"
        duration_model_path = path / duration_model_filename
        duration_predictor = None

        if duration_model_path.exists():
            duration_predictor = DurationPredictor(
                transformer=DurationTransformer(
                    dim=512,
                    depth=12,
                    heads=8,
                    text_dim=512,
                    ff_mult=2,
                    conv_layers=0,
                    text_num_embeds=len(vocab),
                ),
                tokenizer=tokenizer,
            )

        state_dict = safetensors.torch.load_file(duration_model_path.as_posix())
        duration_predictor.load_state_dict(state_dict)

        # model

        model_filename = "model.safetensors"
        model_path = path / model_filename

        model = Nanospeech(
            transformer=DiT(
                dim=512,
                depth=18,
                heads=12,
                text_dim=512,
                ff_mult=2,
                conv_layers=4,
                text_num_embeds=len(vocab),
            ),
            tokenizer=tokenizer,
            vocoder=vocos.decode,
        )

        state_dict = safetensors.torch.load_file(model_path.as_posix())
        model.load_state_dict(state_dict)

        # attach the duration predictor after loading weights so we don't have missing keys

        model._duration_predictor = duration_predictor

        return model
