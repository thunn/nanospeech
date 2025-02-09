"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from functools import lru_cache, partial
import math
from pathlib import Path
from typing import Callable, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from einops.array_api import rearrange, reduce, repeat
import einx

from vocos_mlx import Vocos

from nanospeech.utils import fetch_from_hub

# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(
    t: mx.array["b"],  # noqa: F821
    length: int | None = None,
) -> mx.array["b n"]:  # noqa: F722
    if not exists(length):
        length = t.max().item()

    seq = mx.arange(length)
    return einx.less("n, b -> b n", seq, t)


def mask_from_start_end_indices(
    start: mx.array,
    end: mx.array,
    max_seq_len: int | None = None,
):
    seq = mx.arange(max_seq_len).astype(mx.int32)
    return einx.greater_equal("n, b -> b n", seq, start) & einx.less("n, b -> b n", seq, end)


def mask_from_frac_lengths(
    seq_len: mx.array["b"],  # noqa: F821
    frac_lengths: mx.array["b"],  # noqa: F821
    max_length: int | None = None,
):
    lengths = (frac_lengths * seq_len).astype(mx.int32)
    max_start = seq_len - lengths

    rand = mx.random.uniform(0, 1, frac_lengths.shape)
    start = mx.maximum((max_start * rand).astype(mx.int32), 0)
    end = start + lengths

    return mask_from_start_end_indices(start, end, max_length)


def pad_to_length(t: mx.array, length: int, value=0):
    ndim = t.ndim
    seq_len = t.shape[-1]
    if length > seq_len:
        if ndim == 1:
            t = mx.pad(t, [(0, length - seq_len)], constant_values=value)
        elif ndim == 2:
            t = mx.pad(t, [(0, 0), (0, length - seq_len)], constant_values=value)
        else:
            raise ValueError(f"Unsupported padding dims: {ndim}")
    return t[..., :length]


def pad_sequence(t: mx.array, padding_value=0):
    max_len = max([i.shape[-1] for i in t])
    t = mx.array([pad_to_length(i, max_len, padding_value) for i in t])
    return t


# vocab-based tokenizer


def list_str_to_vocab_tensor(text: list[str], vocab: dict[str, int], padding_value=-1) -> mx.array["b nt"]:  # noqa: F722
    list_tensors = [[vocab.get(c, 0) for c in t] for t in text]
    list_tensors = [mx.array(t) for t in list_tensors]
    text = pad_sequence(list_tensors, padding_value=padding_value)
    return text


# utf-8 tokenizer


def list_str_to_tensor(text: list[str], padding_value=-1) -> mx.array["b nt"]:  # noqa: F722
    list_tensors = [mx.array([*bytes(t, "UTF-8")]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value=padding_value)
    return padded_tensor


# raw wav to mel spec


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: float | None = None,
    norm: str | None = None,
    mel_scale: str = "htk",
) -> mx.array:
    """
    Compute torch-compatible mel filterbanks.

    Args:
        sample_rate: Sampling rate of the audio.
        n_fft: Number of FFT points.
        n_mels: Number of mel bands.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        norm: Normalization mode.
        mel_scale: Mel scale type.

    Returns:
        mx.array of shape (n_mels, n_fft // 2 + 1) containing mel filterbanks.
    """

    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * mx.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    f_max = f_max or sample_rate / 2

    # generate frequency points

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes))

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1)
    return filterbank


@lru_cache(maxsize=None)
def hanning(size):
    """
    Compute the Hanning window.

    Args:
        size: Size of the window.

        Returns:
            mx.array of shape (size,) containing the Hanning window.
    """
    return mx.array(np.hanning(size + 1)[:-1])


def stft(
    x,
    window,
    nperseg=256,
    noverlap=None,
    nfft=None,
    pad_mode="constant",
):
    """
    Compute the short-time Fourier transform of a signal.

    Args:
        x: mx.array of shape (t,) containing the input signal.
        window: mx.array of shape (nperseg,) containing the window function.
        nperseg: Number of samples per segment.
        noverlap: Number of overlapping samples.
        nfft: Number of FFT points.
        pad_mode: Padding mode.

    Returns:
        mx.array of shape (t, nfft // 2 + 1) containing the short-time Fourier transform.
    """
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
):
    """
    Compute log-mel spectrograms for a batch of audio inputs.

    Args:
        audio: mx.array of shape [t] or [b, t] containing audio samples.
        sample_rate: Sampling rate of the audio.
        n_mels: Number of mel bands.
        n_fft: Number of FFT points.
        hop_length: Hop length between frames.
        padding: Amount of padding to add to each audio signal.

    Returns:
        mx.array of shape (batch_size, n_mels, frames) containing log-mel spectrograms.
    """

    if audio.ndim == 1:
        audio = mx.expand_dims(audio, axis=0)

    filters = mel_filters(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, norm=None, mel_scale="htk")

    batch = audio.shape[0]
    outputs = []

    for i in range(batch):
        one_audio = audio[i]

        if padding > 0:
            one_audio = mx.pad(one_audio, (0, padding))

        freqs = stft(one_audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
        magnitudes = mx.abs(freqs[:-1, :])

        mel_spec = mx.matmul(magnitudes, filters.T)
        log_spec = mx.maximum(mel_spec, 1e-5).log()
        outputs.append(log_spec)

    max_seq_len = max([x.shape[1] for x in outputs])
    outputs = [mx.pad(x, (0, max_seq_len - x.shape[1])) for x in outputs]
    return mx.stack(outputs, axis=0)


class MelSpec(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        return log_mel_spectrogram(audio, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)


# RoPE


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.scale = None
            return

        scale = (mx.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.scale = scale

    def forward_from_seq_len(self, seq_len: int) -> tuple[mx.array, float]:
        t = mx.arange(seq_len)
        return self(t)

    def __call__(self, t: mx.array) -> tuple[mx.array, float]:
        max_pos = t.max() + 1

        freqs = mx.einsum("i , j -> i j", t.astype(self._inv_freq.dtype), self._inv_freq) / self.interpolation_factor
        freqs = mx.stack((freqs, freqs), axis=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if self.scale is None:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = mx.stack((scale, scale), axis=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = [mx.squeeze(s, axis=-1) for s in mx.split(x, x.shape[-1], axis=-1)]
    x = mx.stack([-x2, x1], axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]

    freqs = freqs[-seq_len:, :]
    scale = scale[-seq_len:, :] if isinstance(scale, mx.array) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = mx.concatenate((t, t_unrotated), axis=-1)

    return out


# global response normalization


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x):
        Gx = mx.linalg.norm(x, ord=2, axis=1, keepdims=True)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-v2 block


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
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise conv
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
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

    def __call__(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = scale * mx.expand_dims(x, axis=1) * mx.expand_dims(emb, axis=0)
        emb = mx.concatenate([emb.sin(), emb.cos()], axis=-1)
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

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        if mask is not None:
            mask = mask[..., None]
            x = x * mask

        out = self.conv1d(x)

        if mask is not None:
            out = out * mask

        return out


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def __call__(self, timestep: mx.array["b"]) -> mx.array["b d"]:  # noqa: F722 F821
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)
        return time


# feed forward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approx=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def __call__(self, x: mx.array) -> mx.array:
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
        self._scale_factor = 1 / mx.sqrt(dim_head)

        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(dropout))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (
                    xpos_scale,
                    xpos_scale**-1.0,
                )
                if xpos_scale is not None
                else (1.0, 1.0)
            )

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        query = query.reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
        key = key.reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
        value = value.reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(batch, self.heads, 1, seq_len)
        else:
            attn_mask = None

        x = mx.fast.scaled_dot_product_attention(q=query, k=key, v=value, scale=self._scale_factor, mask=attn_mask)
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1).astype(query.dtype)

        # linear proj
        x = self.to_out(x)

        if attn_mask is not None:
            x = x * mask[:, :, None]

        return x


# Text embedding


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim))
    t = mx.arange(end)
    freqs = mx.outer(t, freqs).astype(mx.float32)
    freqs_cos = freqs.cos()  # real part
    freqs_sin = freqs.sin()  # imaginary part
    return mx.concatenate([freqs_cos, freqs_sin], axis=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * mx.ones_like(start)
    pos = mx.expand_dims(start, axis=1) + (mx.expand_dims(mx.arange(length), axis=0) * mx.expand_dims(scale, axis=1)).astype(mx.int32)
    # avoid extra long error.
    pos = mx.where(pos < max_pos, pos, max_pos - 1)
    return pos


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self._freqs_cis = precompute_freqs_cis(text_dim, self.precompute_max_pos)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def __call__(self, text, seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]

        # use 0 as filler token. we rely on text being padded with -1 values.
        text = text + 1

        # curtail if character tokens are more than the mel spec tokens
        text = text[:, :seq_len]

        text = mx.pad(text, [(0, 0), (0, seq_len - text_len)], constant_values=0)

        # cfg for text
        text = mx.where(drop_text, mx.zeros_like(text), text)
        text = self.text_embed(text)  # b n -> b n d

        if self.extra_modeling:
            # sinus pos emb
            batch_start = mx.zeros((batch,), dtype=mx.int32)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self._freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnext v2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: mx.array,  # b n d
        cond: mx.array,  # b n d
        text_embed: mx.array,  # b n d
        drop_audio_cond=False,
    ):
        # cfg for cond audio
        cond = mx.where(drop_audio_cond, mx.zeros_like(cond), cond)
        x = self.proj(mx.concatenate((x, cond, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


# AdaLNZero


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array | None = None) -> mx.array:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(emb, 6, axis=1)

        x = self.norm(x) * (1 + mx.expand_dims(scale_msa, axis=1)) + mx.expand_dims(shift_msa, axis=1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array | None = None) -> mx.array:
        emb = self.linear(self.silu(emb))
        scale, shift = mx.split(emb, 2, axis=1)

        x = self.norm(x) * (1 + mx.expand_dims(scale, axis=1)) + mx.expand_dims(shift, axis=1)
        return x


# diffusion transformer


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.0):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def __call__(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + mx.expand_dims(gate_msa, axis=1) * attn_output
        norm = self.ff_norm(x) * (1 + mx.expand_dims(scale_mlp, axis=1)) + mx.expand_dims(shift_mlp, axis=1)
        ff_output = self.ff(norm)
        x = x + mx.expand_dims(gate_mlp, axis=1) * ff_output

        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.0,
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

        self.transformer_blocks = [
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim, bias=False)
        nn.init.constant(0.0)(self.proj_out.weight)

    def __call__(
        self,
        x: mx.array["b n d"],  # nosied input audio  # noqa: F722
        cond: mx.array["b n d"],  # masked cond audio  # noqa: F722
        text: mx.array["b nt"],  # text  # noqa: F722
        time: mx.array["b"],  # time step  # noqa: F821
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: mx.array["b n"] | None = None,  # noqa: F722
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

# tensor helpers


def maybe_masked_mean(t: mx.array, mask: mx.array | None = None) -> mx.array:
    if not exists(mask):
        return t.mean(dim=1)

    t = einx.where("b n, b n d, -> b n d", mask, t, 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.astype(mx.int32), "b n -> b", "sum")

    return einx.divide("b d, b -> b d", num, mx.maximum(den, 1))


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: mx.array) -> mx.array:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
    ):
        x = self.proj(mx.concatenate((x, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DurationBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def __call__(self, x, mask=None, rope=None):
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
        dropout=0.0,
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

        self.transformer_blocks = [
            DurationBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.norm_out = nn.RMSNorm(dim)

    def __call__(
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
        transformer: DurationTransformer,
        mel_spec_kwargs: dict = dict(),
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        super().__init__()

        self._mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self._mel_spec.n_mels

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.to_pred = nn.Sequential(nn.Linear(transformer.dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ..."))

    def __call__(
        self,
        inp: mx.array["b n d"] | mx.array["b nw"],  # mel or raw wave  # noqa: F722
        text: mx.array | list[str],
        *,
        lens: mx.array["b"] | None = None,  # noqa: F821
        return_loss=False,
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                tokenized_text = self.tokenizer(text)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert tokenized_text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = mx.random.uniform(0, 1, (batch,))
            rand_index = (rand_frac_index * lens).astype(mx.int32)

            seq = mx.arange(seq_len)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending

        inp = mx.where(repeat(mask, "b n -> b n d", d=self.num_channels), inp, mx.zeros_like(inp))

        x = self.transformer(inp, text=tokenized_text)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        if not return_loss:
            return pred

        # loss

        duration = lens.astype(pred.dtype) / SAMPLES_PER_SECOND
        return nn.losses.l1_loss(pred, duration)


# ode solvers


def odeint_euler(func, y0, t):
    """
    Solves ODE using the Euler method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
    - y0: Initial state, an MLX array of any shape.
    - t: Array of time steps, an MLX array.
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

    return mx.stack(ys)


def odeint_midpoint(func, y0, t):
    """
    Solves ODE using the midpoint method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
    - y0: Initial state, an MLX array of any shape.
    - t: Array of time steps, an MLX array.
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

    return mx.stack(ys)


def odeint_rk4(func, y0, t):
    """
    Solves ODE using the Runge-Kutta 4th-order (RK4) method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
    - y0: Initial state, an MLX array of any shape.
    - t: Array of time steps, an MLX array.
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

    return mx.stack(ys)


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
        self._mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self._mel_spec.n_mels

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

    def __call__(
        self,
        inp: mx.array["b n d"] | mx.array["b nw"],  # mel or raw wave  # noqa: F722
        text: mx.array["b nt"] | list[str],  # noqa: F722
        *,
        lens: mx.array["b"] | None = None,  # noqa: F821
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype = *inp.shape[:2], inp.dtype

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                text = self.tokenizer(text)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally

        frac_lengths = mx.random.uniform(*self.frac_lengths_mask, (batch,))
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, seq_len)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # x0 is gaussian noise

        x0 = mx.random.normal(x1.shape)

        # timestep

        time = mx.random.uniform(0, 1, (batch,), dtype=dtype)

        # sample xt

        t = rearrange(time, "b -> b 1 1")
        w = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling

        cond = mx.where(rand_span_mask[..., None], mx.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate

        rand_audio_drop = mx.random.uniform(0, 1, (1,))
        rand_cond_drop = mx.random.uniform(0, 1, (1,))
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

        loss = nn.losses.mse_loss(pred, flow, reduction="none")

        rand_span_mask = repeat(rand_span_mask, "b n -> b n d", d=self.num_channels)
        masked_loss = mx.where(rand_span_mask, loss, mx.zeros_like(loss))
        loss = mx.sum(masked_loss) / mx.maximum(mx.sum(rand_span_mask), 1e-6)

        return loss.mean()

    def predict_duration(
        self,
        cond: mx.array["b n d"],  # noqa: F722
        text: mx.array["b nt"],  # noqa: F722
        speed: float = 1.0,
    ) -> int:
        duration_in_sec = self._duration_predictor(cond, text)
        frame_rate = self._mel_spec.sample_rate // self._mel_spec.hop_length
        duration = (duration_in_sec * frame_rate / speed).astype(mx.int32)
        return duration

    def sample(
        self,
        cond: mx.array["b n d"] | mx.array["b nw"],  # noqa: F722
        text: mx.array["b nt"] | list[str],  # noqa: F722
        duration: int | mx.array["b"] | None = None,  # noqa: F821
        *,
        lens: mx.array["b"] | None = None,  # noqa: F821
        ode_method: Literal["euler", "midpoint", "rk4"] = "rk4",
        steps=8,
        cfg_strength=2.0,
        speed=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
    ) -> tuple[mx.array, mx.array]:
        self.eval()
        
        cond = mx.array(cond)

        # handle raw wave

        if cond.ndim == 2:
            cond = rearrange(cond, "1 n -> n")
            cond = self._mel_spec(cond)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, dtype = *cond.shape[:2], cond.dtype

        # handle text as string

        if isinstance(text, list):
            if exists(self.tokenizer):
                tokenized_text = self.tokenizer(text)
            else:
                assert False, "if text is a list, a tokenizer must be provided"
            assert tokenized_text.shape[0] == batch

        if not exists(lens):
            lens = mx.full((batch,), cond_seq_len, dtype=mx.int32)

        if exists(tokenized_text):
            text_lens = (tokenized_text != -1).sum(axis=-1)
            lens = mx.maximum(text_lens, lens)

        if cond_seq_len < tokenized_text.shape[1]:
            cond_seq_len = tokenized_text.shape[1]
            cond = mx.pad(cond, [(0, 0), (0, cond_seq_len - cond.shape[1]), (0, 0)])

        # duration

        if duration is None and self._duration_predictor is not None:
            duration = self.predict_duration(cond, text, speed).item()
        elif duration is None:
            raise ValueError("Duration must be provided or a duration predictor must be set.")

        cond_mask = lens_to_mask(lens)

        if isinstance(duration, int):
            duration = mx.full((batch,), duration, dtype=dtype)

        assert lens < duration, "duration must be at least as long as the input"

        duration = mx.clip(duration, 0, max_duration)
        max_duration = int(duration.max().item())

        cond = mx.pad(cond, [(0, 0), (0, max_duration - cond_seq_len), (0, 0)])
        cond_mask = mx.pad(
            cond_mask,
            [(0, 0), (0, max_duration - cond_mask.shape[-1])],
            constant_values=False,
        )
        cond_mask = rearrange(cond_mask, "... -> ... 1")

        # at each step, conditioning is fixed

        step_cond = mx.where(cond_mask, cond, mx.zeros_like(cond))

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
                text=tokenized_text,
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
                text=tokenized_text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )
            output = pred + (pred - null_pred) * cfg_strength
            return output

        if ode_method == "midpoint":
            ode_step_fn = odeint_midpoint
        elif ode_method == "euler":
            ode_step_fn = odeint_euler
        elif ode_method == "rk4":
            ode_step_fn = odeint_rk4
        else:
            raise ValueError(f"Unknown method: {ode_method}")

        # noise input

        y0 = []
        for dur in duration:
            if exists(seed):
                mx.random.seed(seed)
            y0.append(mx.random.normal((dur, self.num_channels), dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0)

        t = mx.linspace(0, 1, steps, dtype=step_cond.dtype)
        if exists(sway_sampling_coef):
            t = t + sway_sampling_coef * (mx.cos(mx.pi / 2 * t) - 1 + t)

        fn = mx.compile(fn)
        trajectory = ode_step_fn(fn, y0, t)

        sampled = trajectory[-1]
        out = sampled[:, cond_seq_len:]

        if exists(self.vocoder):
            out = self.vocoder(out)

        mx.eval(out)
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

        vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

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

            weights = mx.load(duration_model_path.as_posix(), format="safetensors")

            new_weights = {}
            for k, v in weights.items():
                # rename layers
                if "mel_spec." in k:
                    continue
                if ".to_out" in k:
                    k = k.replace(".to_out", ".to_out.layers")
                elif "to_pred" in k:
                    k = k.replace("to_pred", "to_pred.layers")
                elif ".text_blocks" in k:
                    k = k.replace(".text_blocks", ".text_blocks.layers")
                elif ".ff.ff.0.0" in k:
                    k = k.replace(".ff.ff.0.0", ".ff.ff.layers.0.layers.0")
                elif ".ff.ff.2" in k:
                    k = k.replace(".ff.ff.2", ".ff.ff.layers.2")
                elif ".time_mlp" in k:
                    k = k.replace(".time_mlp", ".time_mlp.layers")
                elif ".conv1d" in k:
                    k = k.replace(".conv1d", ".conv1d.layers")

                # reshape weights
                if ".dwconv.weight" in k:
                    v = v.swapaxes(1, 2)
                elif ".conv1d.layers.0.weight" in k:
                    v = v.swapaxes(1, 2)
                elif ".conv1d.layers.2.weight" in k:
                    v = v.swapaxes(1, 2)

                new_weights[k] = v

            weights = new_weights

            duration_predictor.load_weights(list(weights.items()))
            mx.eval(duration_predictor.parameters())
        else:
            duration_predictor = None
            print(f"Could not find duration predictor at {duration_model_path}")

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
            duration_predictor=duration_predictor,
            tokenizer=tokenizer,
            vocoder=vocos.decode,
        )

        weights = mx.load(model_path.as_posix(), format="safetensors")

        new_weights = {}
        for k, v in weights.items():
            # rename layers
            if "mel_spec." in k:
                continue
            if ".to_out" in k:
                k = k.replace(".to_out", ".to_out.layers")
            elif ".text_blocks" in k:
                k = k.replace(".text_blocks", ".text_blocks.layers")
            elif ".ff.ff.0.0" in k:
                k = k.replace(".ff.ff.0.0", ".ff.ff.layers.0.layers.0")
            elif ".ff.ff.2" in k:
                k = k.replace(".ff.ff.2", ".ff.ff.layers.2")
            elif ".time_mlp" in k:
                k = k.replace(".time_mlp", ".time_mlp.layers")
            elif ".conv1d" in k:
                k = k.replace(".conv1d", ".conv1d.layers")

            # reshape weights
            if ".dwconv.weight" in k:
                v = v.swapaxes(1, 2)
            elif ".conv1d.layers.0.weight" in k:
                v = v.swapaxes(1, 2)
            elif ".conv1d.layers.2.weight" in k:
                v = v.swapaxes(1, 2)

            new_weights[k] = v

        weights = new_weights

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        return model
