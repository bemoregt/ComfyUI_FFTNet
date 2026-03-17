"""
기존 HuggingFace LLM 의 Self-Attention 레이어를 FFTNetAttentionWrapper 로 교체하는 유틸리티.

지원 모델 계열 (동일한 .self_attn 구조):
  LLaMA, LLaMA-3, Mistral, Qwen2, Falcon, OPT, ...
  (decoder layer 에 self_attn 속성이 있는 모든 모델)
"""

import os
import json
import torch
from .fftnet_attention import FFTNetAttentionWrapper, _decoder_return_count

FFTNET_MARKER = "fftnet_upgraded"   # 체크포인트 타입 식별자


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _get_decoder_layers(model):
    """
    모델에서 decoder layer 리스트를 반환.
    model.model.layers  (LLaMA, Mistral, Qwen2 …)
    model.layers        (일부 소형 모델)
    model.transformer.h (GPT-2 계열)
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "decoder layer 를 찾을 수 없습니다. "
        "model.model.layers / model.layers / model.transformer.h 중 하나여야 합니다."
    )


# ── 공개 API ──────────────────────────────────────────────────────────────────

def replace_with_fftnet(model, window_size: int = 4) -> int:
    """
    모델의 모든 Self-Attention 레이어를 FFTNetAttentionWrapper 로 교체.

    Returns
    -------
    replaced : 교체된 레이어 수
    """
    hidden_size = model.config.hidden_size
    max_seq_len = getattr(model.config, "max_position_embeddings", 2048)

    layers   = _get_decoder_layers(model)
    replaced = 0

    for layer in layers:
        if not hasattr(layer, "self_attn"):
            continue
        ret_count = _decoder_return_count(layer)
        wrapper = FFTNetAttentionWrapper(
            hidden_size  = hidden_size,
            max_seq_len  = max_seq_len,
            window_size  = window_size,
            return_count = ret_count,
        )
        print(f"  layer {replaced}: return_count={ret_count}")
        layer.self_attn = wrapper
        replaced += 1

    print(f"[replace_attention] {replaced} attention layer → FFTNetMixer 교체 완료")
    return replaced


def freeze_non_fftnet(model) -> tuple[int, int]:
    """
    FFTNet 믹서 파라미터만 학습 가능하게 두고 나머지를 freeze.

    Returns
    -------
    (trainable, frozen) : 파라미터 수 쌍
    """
    trainable = frozen = 0
    for name, param in model.named_parameters():
        is_fftnet = "self_attn" in name   # 교체된 레이어는 여전히 self_attn 이름 유지
        param.requires_grad = is_fftnet
        if is_fftnet:
            trainable += param.numel()
        else:
            frozen += param.numel()

    print(
        f"[replace_attention] 학습 파라미터: {trainable:,}  "
        f"| Freeze: {frozen:,}  "
        f"| 비율: {trainable / (trainable + frozen) * 100:.1f}%"
    )
    return trainable, frozen


def save_upgraded_checkpoint(model, tokenizer, output_dir: str, fftnet_config: dict):
    """
    학습 완료된 업그레이드 모델을 저장.

    output_dir/
    ├── fftnet_upgraded.pt     ← state_dict + 설정
    └── tokenizer/             ← HuggingFace 토크나이저 파일
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "type":         FFTNET_MARKER,
        "llama_config": model.config.to_dict(),
        "fftnet_config": fftnet_config,
        "state_dict":   model.state_dict(),
    }
    ckpt_path = os.path.join(output_dir, "fftnet_upgraded.pt")
    torch.save(payload, ckpt_path)
    print(f"[replace_attention] 체크포인트 저장: {ckpt_path}")

    tok_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    print(f"[replace_attention] 토크나이저 저장: {tok_dir}")


def load_upgraded_model(checkpoint_path: str, device: torch.device, window_size: int = 4):
    """
    save_upgraded_checkpoint() 로 저장된 체크포인트를 로드.

    Returns
    -------
    model, tokenizer, fftnet_config
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers import LlamaConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if ckpt.get("type") != FFTNET_MARKER:
        raise ValueError(f"[load_upgraded_model] 이 파일은 {FFTNET_MARKER} 체크포인트가 아닙니다.")

    fftnet_cfg  = ckpt["fftnet_config"]
    llama_cfg_d = ckpt["llama_config"]
    state_dict  = ckpt["state_dict"]

    # ── Config 복원 (Hub 접속 불필요) ────────────────────────────────────────
    model_type = llama_cfg_d.get("model_type", "llama")

    # 저장 시 생긴 내부 키(_name_or_path 등)를 보존하면서 Config 재구성
    try:
        cfg_cls = AutoConfig.for_model(model_type)
        # 모르는 키는 조용히 무시
        import dataclasses, inspect as _inspect
        known = set(_inspect.signature(cfg_cls.__init__).parameters.keys()) - {"self", "kwargs"}
        init_kw = {k: v for k, v in llama_cfg_d.items() if k in known or k.startswith("_")}
        config = cfg_cls(**init_kw)
        # 나머지 속성도 덮어쓰기 (vocab_size 등 중요한 것 포함)
        for k, v in llama_cfg_d.items():
            try:
                setattr(config, k, v)
            except Exception:
                pass
    except Exception as e:
        print(f"[load_upgraded_model] Config 복원 fallback: {e}")
        config = LlamaConfig(**{k: v for k, v in llama_cfg_d.items()
                                if not k.startswith("transformers_version")})

    # ── 모델 구조 생성 (CPU, 실제 텐서) ──────────────────────────────────────
    # torch.device("meta") + to_empty() 방식은 환경에 따라 불안정하므로
    # 직접 CPU 에 빈 모델을 생성 후 state_dict 로 채운다.
    print(f"[load_upgraded_model] 모델 구조 생성 중 ({model_type}) ...")
    model = AutoModelForCausalLM.from_config(config)

    # ── Attention → FFTNet 교체 (가중치 로드 전에 구조를 맞춤) ────────────────
    replace_with_fftnet(model, window_size=fftnet_cfg.get("window_size", window_size))

    # ── 가중치 로드 ───────────────────────────────────────────────────────────
    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if missing:
        print(f"[load_upgraded_model] missing  keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[load_upgraded_model] unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    print(f"[load_upgraded_model] 로드 완료 ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── 토크나이저 ────────────────────────────────────────────────────────────
    tok_dir = os.path.join(os.path.dirname(checkpoint_path), "tokenizer")
    if os.path.isdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        base_name = llama_cfg_d.get("_name_or_path", "")
        if base_name:
            print(f"[load_upgraded_model] 토크나이저를 {base_name} 에서 로드 중 ...")
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        else:
            tokenizer = None
            print("[load_upgraded_model] 토크나이저를 찾을 수 없습니다. tokenizer_path 를 직접 지정하세요.")

    return model, tokenizer, fftnet_cfg
