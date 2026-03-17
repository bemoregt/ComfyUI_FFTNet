"""
ComfyUI custom nodes for FFTNet inference.

Nodes:
  • LoadFFTNetModel  – load a trained FFTNet checkpoint + tokenizer
                       (standalone FFTNetForCausalLM 또는 Upgrade된 LLaMA 모두 지원)
  • FFTNetGenerate   – generate text from a prompt
"""

import os
import json
import threading
import torch
from .fftnet_arch import FFTNetForCausalLM
from .upgrade.replace_attention import FFTNET_MARKER


# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def _load_tokenizer(tokenizer_path: str | None):
    """
    Try to load a tokenizer in order of preference:
      1. transformers AutoTokenizer from model directory
      2. tiktoken (gpt-2 encoding)
      3. Minimal character-level fallback

    Returns an object with .encode(str)->list[int] and .decode(list[int])->str
    """
    # 1) transformers
    if tokenizer_path and os.path.isdir(tokenizer_path):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path)
            return tok
        except Exception as e:
            print(f"[FFTNet] transformers tokenizer failed: {e}")

    # 2) tiktoken (gpt-2)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        class TiktokenWrapper:
            def encode(self, text: str):
                return enc.encode(text)
            def decode(self, ids):
                return enc.decode(ids)

        return TiktokenWrapper()
    except Exception:
        pass

    # 3) Character-level fallback
    class CharTokenizer:
        def encode(self, text: str):
            return [ord(c) for c in text]
        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    print("[FFTNet] Warning: using character-level tokenizer (install tiktoken for better results)")
    return CharTokenizer()


# ── Thread helpers (escape inference_mode) ────────────────────────────────────

def _run_in_thread(fn, *args, **kwargs):
    """ComfyUI wraps node execution in torch.inference_mode(); run in fresh thread to escape."""
    result = [None]
    error  = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if error[0] is not None:
        raise error[0]
    return result[0]


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 – LoadFFTNetModel
# ══════════════════════════════════════════════════════════════════════════════

class LoadFFTNetModel:
    """
    Load a trained FFTNet checkpoint.

    The checkpoint (.pt / .pth) should be a dict saved via:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": { ... },          # optional – model hyper-parameters
            "tokenizer": "path/or/name" # optional
        }, path)

    If the checkpoint contains only the state_dict (no wrapper dict),
    model hyper-parameters must be supplied through this node's inputs.
    A config.json next to the checkpoint is also auto-detected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "/path/to/fftnet_model.pt",
                    "multiline": False,
                }),
                "device": (["auto", "cpu", "cuda", "mps"],),
            },
            "optional": {
                # Override / supply model config manually
                "vocab_size":   ("INT",   {"default": 50257, "min": 256,  "max": 256000}),
                "d_model":      ("INT",   {"default": 512,   "min": 64,   "max": 4096}),
                "n_layers":     ("INT",   {"default": 6,     "min": 1,    "max": 64}),
                "d_ff":         ("INT",   {"default": 2048,  "min": 64,   "max": 16384}),
                "max_seq_len":  ("INT",   {"default": 512,   "min": 64,   "max": 8192}),
                "window_size":  ("INT",   {"default": 4,     "min": 1,    "max": 64}),
                # Optional: directory or path where the tokenizer lives
                "tokenizer_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES  = ("FFTNET_MODEL",)
    RETURN_NAMES  = ("fftnet_model",)
    FUNCTION      = "load_model"
    CATEGORY      = "FFTNet"
    DESCRIPTION   = (
        "학습된 FFTNet 모델 체크포인트를 로드합니다. "
        "체크포인트 파일(.pt/.pth)과 선택적으로 토크나이저 경로를 입력하세요."
    )

    def load_model(
        self,
        checkpoint_path: str,
        device: str,
        vocab_size: int   = 50257,
        d_model:    int   = 512,
        n_layers:   int   = 6,
        d_ff:       int   = 2048,
        max_seq_len: int  = 512,
        window_size: int  = 4,
        tokenizer_path: str = "",
    ):
        return _run_in_thread(
            self._load_inner,
            checkpoint_path, device,
            vocab_size, d_model, n_layers, d_ff, max_seq_len, window_size,
            tokenizer_path,
        )

    def _load_inner(
        self,
        checkpoint_path, device_str,
        vocab_size, d_model, n_layers, d_ff, max_seq_len, window_size,
        tokenizer_path,
    ):
        # ── resolve device ──────────────────────────────────────────────────
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)

        print(f"[FFTNet] Loading checkpoint: {checkpoint_path}  (device={device})")

        # ── load checkpoint ─────────────────────────────────────────────────
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"[FFTNet] Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # ── Upgraded LLaMA 체크포인트 분기 ──────────────────────────────────
        if isinstance(ckpt, dict) and ckpt.get("type") == FFTNET_MARKER:
            return self._load_upgraded_inner(
                ckpt, checkpoint_path, device, tokenizer_path, window_size
            )

        # Support both bare state_dict and wrapped dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            ckpt_config = ckpt.get("config", {})
            ckpt_tok_path = ckpt.get("tokenizer", "")
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
            ckpt_config = {}
            ckpt_tok_path = ""
        else:
            # e.g. full model object
            state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
            ckpt_config = getattr(ckpt, "config", {}) if hasattr(ckpt, "config") else {}
            ckpt_tok_path = ""

        # ── try auto-detect config.json next to checkpoint ──────────────────
        cfg_json = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if os.path.isfile(cfg_json):
            with open(cfg_json) as f:
                file_config = json.load(f)
            print(f"[FFTNet] Loaded config from {cfg_json}")
        else:
            file_config = {}

        # Priority: node inputs > checkpoint config > file config > defaults
        def _pick(key, node_val, default):
            if key in ckpt_config:
                return ckpt_config[key]
            if key in file_config:
                return file_config[key]
            return node_val if node_val != default else default

        config = {
            "vocab_size":    _pick("vocab_size",  vocab_size,  50257),
            "d_model":       _pick("d_model",     d_model,     512),
            "n_layers":      _pick("n_layers",    n_layers,    6),
            "d_ff":          _pick("d_ff",        d_ff,        2048),
            "max_seq_len":   _pick("max_seq_len", max_seq_len, 512),
            "window_size":   _pick("window_size", window_size, 4),
            "dropout":       ckpt_config.get("dropout", file_config.get("dropout", 0.0)),
            "eos_token_id":  ckpt_config.get("eos_token_id", file_config.get("eos_token_id", None)),
        }
        print(f"[FFTNet] Model config: {config}")

        # ── build model & load weights ──────────────────────────────────────
        model = FFTNetForCausalLM(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FFTNet] Warning – missing keys: {missing[:5]} ...")
        if unexpected:
            print(f"[FFTNet] Warning – unexpected keys: {unexpected[:5]} ...")
        model.to(device)
        model.eval()
        print(f"[FFTNet] Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

        # ── tokenizer ───────────────────────────────────────────────────────
        tok_search = tokenizer_path or ckpt_tok_path or os.path.dirname(checkpoint_path)
        tokenizer  = _load_tokenizer(tok_search if tok_search else None)
        print(f"[FFTNet] Tokenizer: {type(tokenizer).__name__}")

        bundle = {
            "model":     model,
            "tokenizer": tokenizer,
            "device":    device,
            "config":    config,
            "model_type": "standalone",
        }
        return (bundle,)

    def _load_upgraded_inner(self, ckpt, checkpoint_path, device, tokenizer_path, window_size):
        """Upgrade된 LLaMA 체크포인트 로드 (transformers 필요)."""
        from .upgrade.replace_attention import load_upgraded_model

        ws = ckpt.get("fftnet_config", {}).get("window_size", window_size)
        model, tokenizer, fftnet_cfg = load_upgraded_model(checkpoint_path, device, window_size=ws)

        # 노드에서 별도로 tokenizer_path 를 지정한 경우 우선 적용
        if tokenizer_path:
            alt = _load_tokenizer(tokenizer_path)
            if alt is not None:
                tokenizer = alt

        if tokenizer is None:
            tokenizer = _load_tokenizer(None)

        cfg = ckpt.get("fftnet_config", {})
        bundle = {
            "model":      model,
            "tokenizer":  tokenizer,
            "device":     device,
            "config":     cfg,
            "model_type": "upgraded",
        }
        print(f"[FFTNet] Upgraded LLaMA 로드 완료 (window_size={ws})")
        return (bundle,)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 – FFTNetGenerate
# ══════════════════════════════════════════════════════════════════════════════

class FFTNetGenerate:
    """
    Run autoregressive text generation with a loaded FFTNet model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fftnet_model": ("FFTNET_MODEL",),
                "prompt": ("STRING", {
                    "default": "Once upon a time",
                    "multiline": True,
                }),
                "max_new_tokens": ("INT",   {"default": 200,  "min": 1,    "max": 4096}),
                "temperature":    ("FLOAT", {"default": 0.8,  "min": 0.01, "max": 2.0,  "step": 0.01}),
                "top_k":          ("INT",   {"default": 50,   "min": 0,    "max": 1000}),
                "top_p":          ("FLOAT", {"default": 0.9,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("generated_text",)
    OUTPUT_NODE   = True
    FUNCTION      = "generate"
    CATEGORY      = "FFTNet"
    DESCRIPTION   = (
        "FFTNet 모델로 텍스트를 생성합니다. "
        "프롬프트를 입력하면 모델이 이어지는 텍스트를 자기회귀적으로 생성합니다."
    )

    def generate(
        self,
        fftnet_model: dict,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        system_prompt: str = "",
    ):
        return _run_in_thread(
            self._generate_inner,
            fftnet_model, prompt, max_new_tokens,
            temperature, top_k, top_p, repetition_penalty,
            seed, system_prompt,
        )

    def _generate_inner(
        self,
        bundle: dict,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        system_prompt: str,
    ):
        model      = bundle["model"]
        tokenizer  = bundle["tokenizer"]
        device     = bundle["device"]
        model_type = bundle.get("model_type", "standalone")

        if seed >= 0:
            torch.manual_seed(seed)

        full_text = f"{system_prompt.strip()}\n\n{prompt}" if system_prompt.strip() else prompt

        # ── tokenize ─────────────────────────────────────────────────────────
        token_ids = tokenizer.encode(full_text)
        if hasattr(token_ids, "input_ids"):
            ids = token_ids.input_ids
            token_ids = ids[0].tolist() if isinstance(ids, torch.Tensor) else list(ids)
        elif isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze().tolist()

        input_ids  = torch.tensor([token_ids], dtype=torch.long, device=device)
        prompt_len = input_ids.shape[1]
        print(f"[FFTNet] [{model_type}] Prompt tokens: {prompt_len} | max_new: {max_new_tokens}")

        # ── generate ─────────────────────────────────────────────────────────
        if model_type == "upgraded":
            # HuggingFace model.generate() 사용
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            # standalone FFTNetForCausalLM.generate()
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # ── decode ───────────────────────────────────────────────────────────
        new_ids = output_ids[0, prompt_len:].tolist()
        try:
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        except TypeError:
            decoded = tokenizer.decode(new_ids)

        print(f"[FFTNet] Generated {len(new_ids)} tokens.\n{decoded}")
        return (decoded,)
