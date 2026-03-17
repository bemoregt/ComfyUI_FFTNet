# ComfyUI_FFTNet

ComfyUI 커스텀 노드 — **FFTNet** 언어 모델 추론.

FFTNet은 2025년 2월 발표된 논문 ["The FFT Strikes Back: An Efficient Alternative to Self-Attention"](https://arxiv.org/abs/2502.18394) (arXiv:2502.18394)에서 제안된 모델로, 기존 트랜스포머의 O(n²) 셀프어텐션을 **O(n log n) FFT 기반 전역 토큰 믹싱**으로 대체합니다.

![ComfyUI FFTNet 워크플로우](ScrShot%206.png)

---

## 아키텍처 개요

```
입력 토큰
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼  (× n_layers)
┌─────────────────────────────────────┐
│  LayerNorm                          │
│       │                             │
│  ┌────┴─────┐   ┌──────────────┐   │
│  │ Global   │   │ Local Window │   │
│  │ FFT 브랜치│   │ Conv 브랜치   │   │
│  │          │   │              │   │
│  │ rfft     │   │ Depthwise    │   │
│  │ ×학습필터 │   │ Conv1d       │   │
│  │ modReLU  │   │              │   │
│  │ irfft    │   │              │   │
│  └────┬─────┘   └──────┬───────┘   │
│       └────── Gate ────┘           │
│                 │                  │
│              Linear                │
│  + Residual                        │
│                                    │
│  LayerNorm → FFN (GELU) → Residual │
└─────────────────────────────────────┘
    │
    ▼
LayerNorm → LM Head (vocab 크기)
```

| 컴포넌트 | 설명 |
|---|---|
| **SpectralFilter** | rfft → 학습 가능한 복소수 필터 곱 → modReLU → irfft |
| **modReLU** | 주파수 도메인 활성화: `ReLU(|z| + b) * z / |z|` |
| **LocalWindowMixing** | Depthwise Conv1d로 단거리 의존성 포착 |
| **FFTNetMixer** | 전역/지역 브랜치를 Sigmoid 게이트로 결합 |
| **FFTNetBlock** | Pre-Norm + Mixer + Pre-Norm + FFN (트랜스포머 블록) |

---

## 설치

```bash
# 1. 리포지토리 클론 (또는 이미 존재하는 경우 생략)
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI_FFTNet

# 2. 의존성 설치
pip install torch>=2.0.0
pip install tiktoken          # 권장: GPT-2 토크나이저
# pip install transformers    # 선택: HuggingFace 토크나이저 지원
```

또는 이미 `/Users/m1_4k/ComfyUI_FFTNet/`에 있는 경우 `custom_nodes`에 심볼릭 링크가 자동으로 설정되어 있습니다.

---

## 노드

### Load FFTNet Model

학습된 FFTNet 체크포인트를 로드합니다.

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `checkpoint_path` | STRING | `.pt` / `.pth` 파일 경로 |
| `device` | 선택 | `auto` / `cpu` / `cuda` / `mps` |
| `vocab_size` | INT | 어휘 크기 (기본값: 50257) |
| `d_model` | INT | 임베딩 차원 (기본값: 512) |
| `n_layers` | INT | FFTNet 블록 수 (기본값: 6) |
| `d_ff` | INT | FFN 내부 차원 (기본값: 2048) |
| `max_seq_len` | INT | 최대 시퀀스 길이 (기본값: 512) |
| `window_size` | INT | 로컬 Conv 커널 크기 (기본값: 4) |
| `tokenizer_path` | STRING | 토크나이저 디렉토리 (선택) |

**출력:** `FFTNET_MODEL`

> 체크포인트 옆에 `config.json`이 있으면 모델 설정이 자동으로 읽힙니다.
> 노드 입력값 > 체크포인트 내부 config > config.json > 기본값 순으로 우선순위가 적용됩니다.

---

### FFTNet Generate

FFTNet 모델로 텍스트를 자기회귀적으로 생성합니다.

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `fftnet_model` | FFTNET_MODEL | Load FFTNet Model의 출력 |
| `prompt` | STRING | 입력 프롬프트 |
| `max_new_tokens` | INT | 최대 생성 토큰 수 (기본값: 200) |
| `temperature` | FLOAT | 샘플링 온도. 낮을수록 확정적 (기본값: 0.8) |
| `top_k` | INT | Top-K 샘플링. 0이면 비활성화 (기본값: 50) |
| `top_p` | FLOAT | Nucleus 샘플링 임계값 (기본값: 0.9) |
| `repetition_penalty` | FLOAT | 반복 억제 계수 (기본값: 1.1) |
| `seed` | INT | 재현용 시드. -1이면 랜덤 (기본값: -1) |
| `system_prompt` | STRING | 시스템 프롬프트 (선택, 프롬프트 앞에 추가됨) |

**출력:** `STRING` (생성된 텍스트)

---

## 체크포인트 저장 형식

모델 학습 후 아래 형식으로 저장하면 자동으로 모든 설정이 불려옵니다.

```python
import torch

torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size":   50257,
        "d_model":      512,
        "n_layers":     6,
        "d_ff":         2048,
        "max_seq_len":  512,
        "window_size":  4,
        "dropout":      0.0,
        "eos_token_id": 50256,   # GPT-2 <|endoftext|>
    },
}, "fftnet_checkpoint.pt")
```

또는 체크포인트와 같은 디렉토리에 `config.json`을 함께 두어도 됩니다.

```json
{
    "vocab_size": 50257,
    "d_model": 512,
    "n_layers": 6,
    "d_ff": 2048,
    "max_seq_len": 512,
    "window_size": 4,
    "eos_token_id": 50256
}
```

---

## 토크나이저

다음 순서로 자동 선택됩니다.

1. **HuggingFace transformers** — `tokenizer_path`가 유효한 디렉토리일 때
   (`transformers` 라이브러리 필요)
2. **tiktoken (GPT-2)** — `pip install tiktoken`으로 설치된 경우
3. **문자 단위 fallback** — 위 둘 다 없을 때 (어휘가 제한적이므로 권장하지 않음)

---

## 워크플로우 예시

```
[Load FFTNet Model]
  checkpoint_path: /models/fftnet_v1.pt
  device: auto
        │
        ▼  FFTNET_MODEL
[FFTNet Generate]
  prompt: "다음 질문에 답해줘: 대한민국의 수도는?"
  max_new_tokens: 100
  temperature: 0.7
        │
        ▼  STRING
[Show Text / 다른 노드로 연결]
```

---

## 파일 구조

```
ComfyUI_FFTNet/
├── __init__.py        # 노드 등록
├── fftnet_arch.py     # FFTNet 모델 클래스 (ModReLU, SpectralFilter, ...)
├── nodes.py           # ComfyUI 노드 정의
├── requirements.txt
└── README.md
```

---

## 참고 문헌

- Jacob Fein-Ashley et al., **"The FFT Strikes Back: An Efficient Alternative to Self-Attention"**, arXiv:2502.18394 (2025)
  https://arxiv.org/abs/2502.18394
