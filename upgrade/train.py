"""
FFTNet Upgrade 파인튜닝 스크립트
=================================
기존 sLLM (LLaMA, Mistral, Qwen2 …) 의 Self-Attention 을 FFTNet 으로 교체하고
새로 추가된 FFT 파라미터만 파인튜닝합니다.

사용 예시
---------
# Ollama 에서 뽑아낸 / HuggingFace 에서 받은 모델로 MPS 파인튜닝
python -m ComfyUI_FFTNet.upgrade.train \\
    --base_model meta-llama/Llama-3.2-1B \\
    --output_dir ./output/fftnet_llama_1b \\
    --dataset wikitext \\
    --epochs 3 \\
    --batch_size 2 \\
    --seq_len 512 \\
    --lr 3e-4 \\
    --device mps

# 로컬 텍스트 파일로 학습
python -m ComfyUI_FFTNet.upgrade.train \\
    --base_model ./my_llama_dir \\
    --output_dir ./output/fftnet_custom \\
    --data_file ./my_corpus.txt \\
    --epochs 1 \\
    --device mps

# JSONL 파일 ({"text": "..."} 형식)
python -m ComfyUI_FFTNet.upgrade.train \\
    --base_model meta-llama/Llama-3.2-1B \\
    --output_dir ./output/fftnet_llama \\
    --data_file ./data.jsonl \\
    --device mps

Ollama 모델 변환 방법
---------------------
Ollama 는 GGUF 포맷을 사용합니다. 아래 명령으로 HuggingFace 포맷으로 변환:
    ollama pull llama3.2:1b
    # Ollama 캐시 경로: ~/.ollama/models/
    # llama.cpp 의 convert_hf_to_gguf.py 역방향 또는 직접 HuggingFace 에서 다운로드
    huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./llama_3_2_1b
"""

import argparse
import os
import sys
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 직접 실행(python upgrade/train.py) 시 상대 임포트 대응 ───────────────────
# python -m ComfyUI_FFTNet.upgrade.train 이 아닌 경우 sys.path 를 보정한다.
if __package__ is None or __package__ == "":
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    # 이후 from upgrade.xxx 로 임포트
    from upgrade.replace_attention import replace_with_fftnet, freeze_non_fftnet, save_upgraded_checkpoint
else:
    from .replace_attention import replace_with_fftnet, freeze_non_fftnet, save_upgraded_checkpoint


# ── CLI 인자 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FFTNet Upgrade Fine-tuning")
    p.add_argument("--base_model",  required=True,
                   help="HuggingFace 모델 ID 또는 로컬 디렉토리 경로")
    p.add_argument("--output_dir",  required=True,
                   help="체크포인트 저장 디렉토리")
    p.add_argument("--device",      default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])

    # 데이터
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--dataset",   default=None,
                     choices=["wikitext", "openwebtext"],
                     help="HuggingFace datasets 에서 자동 다운로드")
    grp.add_argument("--data_file", default=None,
                     help=".txt 또는 .jsonl 파일 경로")

    # FFTNet 하이퍼파라미터
    p.add_argument("--window_size", type=int, default=4,
                   help="LocalWindowMixing 커널 크기 (기본 4)")
    p.add_argument("--freeze_base", action="store_true", default=True,
                   help="기존 모델 파라미터 freeze (기본 True)")
    p.add_argument("--no_freeze",   action="store_true",
                   help="전체 파라미터 학습 (메모리 많이 필요)")

    # 학습 설정
    p.add_argument("--seq_len",     type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup_steps",type=int,   default=100)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--save_every",  type=int,   default=500,
                   help="N 스텝마다 중간 체크포인트 저장")
    p.add_argument("--dtype",       default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


# ── 디바이스 설정 ─────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    텍스트를 토큰화한 뒤 seq_len 길이 청크로 나눈 데이터셋.
    마지막 토큰 예측 (Causal LM) 용.
    """
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # seq_len+1 길이 청크 (input + label shift)
        n       = (len(token_ids) - 1) // seq_len
        self.data = torch.tensor(token_ids[:n * seq_len + 1], dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start     : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def _load_hf_dataset(name: str, tokenizer, seq_len: int) -> TokenDataset:
    from datasets import load_dataset
    print(f"[train] HuggingFace datasets '{name}' 로드 중 ...")
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = "\n\n".join(ds["text"])
    elif name == "openwebtext":
        ds = load_dataset("openwebtext", split="train[:1%]")  # 1% 샘플
        texts = "\n\n".join(ds["text"])
    else:
        raise ValueError(f"알 수 없는 dataset: {name}")

    print("[train] 토크나이즈 중 ...")
    ids = tokenizer.encode(texts)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids[0].tolist() if isinstance(ids.input_ids, torch.Tensor) else list(ids.input_ids)
    print(f"[train] 총 {len(ids):,} 토큰")
    return TokenDataset(ids, seq_len)


def _load_file_dataset(path: str, tokenizer, seq_len: int) -> TokenDataset:
    import json
    ext = os.path.splitext(path)[1].lower()
    print(f"[train] 파일 로드: {path}")
    if ext == ".jsonl":
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line).get("text", ""))
        text = "\n\n".join(texts)
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()

    print("[train] 토크나이즈 중 ...")
    ids = tokenizer.encode(text)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids[0].tolist() if isinstance(ids.input_ids, torch.Tensor) else list(ids.input_ids)
    elif isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    print(f"[train] 총 {len(ids):,} 토큰")
    return TokenDataset(ids, seq_len)


# ── 학습률 스케줄러 (cosine with warmup) ─────────────────────────────────────

def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype  = {"float32": torch.float32,
              "bfloat16": torch.bfloat16,
              "float16": torch.float16}[args.dtype]

    # MPS + bfloat16 조합 경고
    if device.type == "mps" and dtype == torch.float16:
        print("[train] Warning: MPS 에서 float16 불안정할 수 있습니다. bfloat16 권장.")

    print(f"[train] Device: {device} | dtype: {args.dtype}")

    # ── 상대경로 → 절대경로 변환 (HuggingFace 가 './' 를 거부하므로) ──────────
    if os.path.exists(args.base_model):
        args.base_model = os.path.abspath(args.base_model)
    if args.data_file:
        args.data_file = os.path.abspath(args.data_file)
    args.output_dir = os.path.abspath(args.output_dir)

    # ── 베이스 모델 로드 ──────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[train] 베이스 모델 로드: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # ── Attention → FFTNet 교체 ───────────────────────────────────────────────
    replace_with_fftnet(model, window_size=args.window_size)

    if not args.no_freeze:
        trainable, frozen = freeze_non_fftnet(model)
    else:
        trainable = sum(p.numel() for p in model.parameters())
        frozen    = 0
        print(f"[train] 전체 파인튜닝 모드 | 파라미터: {trainable:,}")

    model = model.to(device)

    # ── 데이터셋 준비 ─────────────────────────────────────────────────────────
    if args.data_file:
        dataset = _load_file_dataset(args.data_file, tokenizer, args.seq_len)
    elif args.dataset:
        dataset = _load_hf_dataset(args.dataset, tokenizer, args.seq_len)
    else:
        print("[train] --dataset 또는 --data_file 을 지정하지 않아 wikitext-2 를 기본 사용합니다.")
        dataset = _load_hf_dataset("wikitext", tokenizer, args.seq_len)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,   # MPS 에서 num_workers > 0 은 불안정
    )
    print(f"[train] 샘플 수: {len(dataset):,} | 배치 수: {len(loader):,}")

    # ── 옵티마이저 / 스케줄러 ─────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    total_steps  = len(loader) * args.epochs
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup_steps, total_steps),
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    print(f"\n{'='*60}")
    print(f"  FFTNet Upgrade 파인튜닝 시작")
    print(f"  베이스 모델  : {args.base_model}")
    print(f"  디바이스     : {device}")
    print(f"  학습 파라미터: {trainable:,}")
    print(f"  총 스텝      : {total_steps:,}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)

            # MPS 에서 bfloat16 일 때 일부 op 가 CPU fallback 발생할 수 있음
            logits = model(input_ids=x).logits       # [B, L, V]
            loss   = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)

            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss  += loss.item()

            # 로그
            if global_step % 50 == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg, 10))
                msg = (
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {global_step:>6} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                    f"LR {lr:.2e}"
                )
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

            # 중간 저장
            if args.save_every > 0 and global_step % args.save_every == 0:
                _save(model, tokenizer, args, global_step)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(loader)
        print(f"\n[Epoch {epoch}] 평균 Loss: {avg_loss:.4f} | "
              f"PPL: {math.exp(min(avg_loss, 10)):.1f} | "
              f"소요: {elapsed:.0f}s\n")

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    _save(model, tokenizer, args, global_step, final=True)
    print(f"\n[train] 완료! 저장 위치: {args.output_dir}")
    print("[train] ComfyUI 에서 Load FFTNet Model 노드로 로드하세요.")
    print(f"        checkpoint_path: {os.path.join(args.output_dir, 'fftnet_upgraded.pt')}")


def _save(model, tokenizer, args, step: int, final: bool = False):
    tag = "final" if final else f"step{step}"
    out = args.output_dir if final else os.path.join(args.output_dir, f"ckpt_{tag}")

    fftnet_config = {
        "window_size": args.window_size,
        "max_seq_len": getattr(model.config, "max_position_embeddings", 2048),
        "base_model":  args.base_model,
        "step":        step,
    }
    save_upgraded_checkpoint(model, tokenizer, out, fftnet_config)


if __name__ == "__main__":
    main()
