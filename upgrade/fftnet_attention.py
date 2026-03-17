"""
FFTNetAttentionWrapper – drop-in replacement for any HuggingFace attention module.

MPS 주의사항
-----------
torch.fft.rfft/irfft 는 MPS 백엔드에서 미지원 (ComplexFloat 미지원).
SpectralFilter.forward()에서 FFT 연산만 자동으로 CPU 로 오프로드한 뒤
결과를 원래 device 로 복귀합니다.  나머지 연산(gate, proj, conv)은 MPS 에서 실행됩니다.
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── modReLU (복소수 도메인 활성화) ───────────────────────────────────────────

class ModReLU(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs()
        return F.relu(mag + self.bias) * z / (mag + 1e-8)


# ── Spectral Filter (MPS-safe) ───────────────────────────────────────────────

class SpectralFilter(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        freq_bins = max_seq_len // 2 + 1
        self.filter_real = nn.Parameter(torch.randn(freq_bins, d_model) * 0.02)
        self.filter_imag = nn.Parameter(torch.zeros(freq_bins, d_model))
        self.modrelu = ModReLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        dev = x.device

        # FFT 는 MPS 미지원 → CPU 로 오프로드
        if dev.type == "mps":
            x_cpu   = x.detach().cpu().requires_grad_(x.requires_grad)
            fr_cpu  = self.filter_real.cpu()
            fi_cpu  = self.filter_imag.cpu()
        else:
            x_cpu, fr_cpu, fi_cpu = x, self.filter_real, self.filter_imag

        x_freq   = torch.fft.rfft(x_cpu, dim=1)               # [B, freq, D]
        freq_len = x_freq.shape[1]
        filt     = torch.complex(fr_cpu[:freq_len], fi_cpu[:freq_len])

        # modReLU bias 도 같은 device 로
        bias_cpu = self.modrelu.bias.to(x_cpu.device)
        mag      = (x_freq * filt.unsqueeze(0)).abs()
        x_filt   = F.relu(mag + bias_cpu) * (x_freq * filt.unsqueeze(0)) / (mag + 1e-8)

        out_cpu  = torch.fft.irfft(x_filt, n=L, dim=1)        # [B, L, D]
        return out_cpu.to(dev)


# ── Local Window Branch ───────────────────────────────────────────────────────

class LocalWindowMixing(nn.Module):
    def __init__(self, d_model: int, window_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=window_size,
            padding=window_size - 1,
            groups=d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L   = x.shape[1]
        out = self.conv(x.transpose(1, 2))[:, :, :L]
        return out.transpose(1, 2)


# ── FFTNet Mixer ─────────────────────────────────────────────────────────────

class FFTNetMixer(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048, window_size: int = 4):
        super().__init__()
        self.spectral = SpectralFilter(d_model, max_seq_len)
        self.local    = LocalWindowMixing(d_model, window_size)
        self.gate     = nn.Linear(d_model * 2, d_model)
        self.proj     = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g     = self.spectral(x)
        l     = self.local(x)
        alpha = torch.sigmoid(self.gate(torch.cat([g, l], dim=-1)))
        return self.proj(alpha * g + (1.0 - alpha) * l)


# ── HuggingFace Attention Drop-in ─────────────────────────────────────────────

def _decoder_return_count(layer) -> int:
    """
    디코더 레이어의 forward 소스를 분석해 self_attn 이 반환해야 할 값의 개수를 반환.

    반환값:
        1  → hidden_states = self.self_attn(...)          (신버전, 텐서 직접)
        2  → hidden_states, present_kv = self.self_attn(...)
        3  → hidden_states, weights, present_kv = self.self_attn(...)  (구버전)
    기본: 3 (불확실할 때 가장 안전한 값)
    """
    import re
    try:
        src = inspect.getsource(type(layer).forward)

        # self.self_attn( 호출 라인에서 좌변 변수 개수 세기
        m = re.search(
            r'([\w\s,]+?)\s*=\s*self\.self_attn\s*\(',
            src,
        )
        if m:
            lhs = m.group(1).strip()
            count = len([v.strip() for v in lhs.split(',') if v.strip()])
            if count in (1, 2, 3):
                return count

        # 인덱스 접근 패턴: result = self.self_attn(...); x = result[0]
        if re.search(r'=\s*self\.self_attn\s*\(', src) and '[0]' in src:
            return 2   # result[0], result[1] 형태로 2개 이상 사용

    except Exception:
        pass

    return 3  # 불확실 → 가장 흔한 구버전 형식


# 하위 호환 alias
def _decoder_expects_tuple(layer) -> bool:
    return _decoder_return_count(layer) > 1


class FFTNetAttentionWrapper(nn.Module):
    """
    기존 Self-Attention 을 완전히 대체하는 FFTNet 믹서.

    Parameters
    ----------
    hidden_size   : 모델 hidden dim (LlamaConfig.hidden_size)
    max_seq_len   : 최대 시퀀스 길이 (LlamaConfig.max_position_embeddings)
    window_size   : 로컬 Conv 커널 크기 (기본 4)
    return_count  : 반환값 개수
                    1 → tensor 직접 (신버전)
                    2 → (tensor, None)          Qwen2 등
                    3 → (tensor, None, None)    구버전 LLaMA/Mistral
    """

    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int = 2048,
        window_size: int = 4,
        return_count: int = 3,
    ):
        super().__init__()
        self.mixer        = FFTNetMixer(hidden_size, max_seq_len, window_size)
        self.return_count = return_count

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # attention_mask, position_ids, past_key_values 등 모두 무시
        out = self.mixer(hidden_states)
        if self.return_count == 2:
            return out, None
        if self.return_count == 3:
            return out, None, None
        return out   # return_count == 1
