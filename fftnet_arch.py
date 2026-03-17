"""
FFTNet architecture – "The FFT Strikes Back: An Efficient Alternative to Self-Attention"
arXiv:2502.18394 (February 2025)

Replaces quadratic self-attention with O(n log n) FFT-based global token mixing
combined with a local windowed convolution branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── modReLU ───────────────────────────────────────────────────────────────────

class ModReLU(nn.Module):
    """
    modReLU in the frequency domain.
    Applies a learnable bias to the magnitude then gates with ReLU,
    preserving the original phase: out = ReLU(|z| + b) * z / (|z| + eps)
    """
    def __init__(self, size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs()
        activated = F.relu(magnitude + self.bias)
        return activated * z / (magnitude + 1e-8)


# ── Spectral Filter ───────────────────────────────────────────────────────────

class SpectralFilter(nn.Module):
    """
    Adaptive learnable spectral filter.
    Transforms tokens to frequency domain via rfft, applies element-wise
    complex multiplication with trainable weights, passes through modReLU,
    then maps back via irfft.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        freq_bins = max_seq_len // 2 + 1
        # Learnable complex filter: stored as real + imag components
        self.filter_real = nn.Parameter(torch.randn(freq_bins, d_model) * 0.02)
        self.filter_imag = nn.Parameter(torch.zeros(freq_bins, d_model))
        self.modrelu = ModReLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)           # [B, freq_bins, D]

        freq_len = x_freq.shape[1]
        filt = torch.complex(
            self.filter_real[:freq_len],             # [freq_len, D]
            self.filter_imag[:freq_len],
        )
        x_filtered = x_freq * filt.unsqueeze(0)     # broadcast over batch
        x_filtered = self.modrelu(x_filtered)
        return torch.fft.irfft(x_filtered, n=L, dim=1)  # [B, L, D]


# ── Local Window Branch ───────────────────────────────────────────────────────

class LocalWindowMixing(nn.Module):
    """
    Depthwise conv captures short-range (local window) dependencies to
    complement the global FFT branch.
    """
    def __init__(self, d_model: int, window_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=window_size,
            padding=window_size - 1,   # causal-style left padding
            groups=d_model,
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.shape[1]
        out = self.conv(x.transpose(1, 2))  # [B, D, L + window - 1]
        out = out[:, :, :L]                 # keep only valid positions
        return out.transpose(1, 2)


# ── FFTNet Token Mixer ─────────────────────────────────────────────────────────

class FFTNetMixer(nn.Module):
    """
    Combines global (FFT) and local (window conv) branches via a learned gate.
    Output is projected back to d_model.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, window_size: int = 4):
        super().__init__()
        self.spectral = SpectralFilter(d_model, max_seq_len)
        self.local    = LocalWindowMixing(d_model, window_size)
        self.gate     = nn.Linear(d_model * 2, d_model)
        self.proj     = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.spectral(x)
        l = self.local(x)
        alpha = torch.sigmoid(self.gate(torch.cat([g, l], dim=-1)))
        mixed = alpha * g + (1.0 - alpha) * l
        return self.proj(mixed)


# ── FFTNet Transformer Block ──────────────────────────────────────────────────

class FFTNetBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        max_seq_len: int = 512,
        window_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.mixer  = FFTNetMixer(d_model, max_seq_len, window_size)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Full Causal LM ────────────────────────────────────────────────────────────

class FFTNetForCausalLM(nn.Module):
    """
    Autoregressive language model built on FFTNet blocks.

    Expected config keys (all optional, defaults shown):
        vocab_size   (int)   50257
        d_model      (int)   512
        n_layers     (int)   6
        d_ff         (int)   d_model * 4
        max_seq_len  (int)   512
        window_size  (int)   4
        dropout      (float) 0.1
        eos_token_id (int)   None
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        vocab_size  = config.get("vocab_size",  50257)
        d_model     = config.get("d_model",     512)
        n_layers    = config.get("n_layers",    6)
        d_ff        = config.get("d_ff",        d_model * 4)
        max_seq_len = config.get("max_seq_len", 512)
        window_size = config.get("window_size", 4)
        dropout     = config.get("dropout",     0.1)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            FFTNetBlock(d_model, d_ff, max_seq_len, window_size, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (output proj shares token embedding matrix)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        max_seq_len = self.config.get("max_seq_len", 512)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).clamp(max=max_seq_len - 1)
        x   = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))   # [B, L, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        generated  = input_ids.clone()
        max_seq_len = self.config.get("max_seq_len", 512)
        eos_id      = self.config.get("eos_token_id", None)

        for _ in range(max_new_tokens):
            ctx   = generated[:, -max_seq_len:]
            logits = self(ctx)[:, -1, :]          # [B, vocab]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for tok in generated[b].unique():
                        logits[b, tok] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                kth, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < kth[:, [-1]]] = -float("inf")

            # Top-p (nucleus)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
                sorted_logits[remove_mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated
