"""
main.py — BoaConstrictor CLI (rewritten for MinGRU backbone)

Key changes from original:
  - Uses model_mingru.py (MinGRU backbone) instead of model.py (Mamba)
  - No mamba_ssm / causal-conv1d / mambapy dependencies needed
  - model_path must be EXPLICITLY provided via CLI or YAML to skip training
  - Auto-checkpoint detection removed — no silent training skips
  - --force-train flag overrides model_path and retrains from scratch
  - Removed networkx import (unused in original)
  - Cleaner timings reporting
"""

import argparse
import os
import time
from pathlib import Path

import yaml
import numpy as np
import torch

from model_mingru import BoaConstrictor, ByteDataloader, make_splits
from boa import BOA
from train import train


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(
    config_arg: str,
    experiments_root: Path = Path("experiments"),
) -> Path:
    """
    Resolve --config argument to an actual .yaml file.
    Order:
      1. Direct file path if it exists
      2. experiments/<name>/<name>.yaml
      3. configs/<name>.yaml
    """
    if config_arg is None:
        return None
    p = Path(config_arg)
    if p.exists():
        return p
    name = p.stem
    exp_cfg = experiments_root / name / f"{name}.yaml"
    if exp_cfg.exists():
        return exp_cfg
    cfg_cfg = Path("configs") / f"{name}.yaml"
    if cfg_cfg.exists():
        return cfg_cfg
    raise FileNotFoundError(
        f"Could not resolve config argument '{config_arg}' to a config file"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run BoaConstrictor experiments from a config file"
    )
    p.add_argument("--config", "-c", type=str, required=False,
                   help="Path to YAML experiment config or experiment name")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable progress bars")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device override (cpu|cuda)")
    p.add_argument("--precision", type=str, default=None,
                   choices=["fp32", "fp16", "fp8"],
                   help="Precision override (training only)")
    p.add_argument("--new-experiment", action="store_true",
                   help="Create a new experiment config interactively")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--compress-only", action="store_true")
    p.add_argument("--decompress-only", action="store_true")
    p.add_argument("--show-timings", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="Verify decompressed bytes match original")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--comparison-baseline-only", action="store_true",
                   help="Run LZMA/ZLIB baselines and exit")
    p.add_argument("--model-path", type=str, default=None,
                   help="Explicit path to a .pt checkpoint — skips training")
    p.add_argument("--force-train", action="store_true",
                   help="Force training even if model_path is set in YAML")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_path(model, path: Path):
    """Load state_dict or full model from a .pt file."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # Try direct state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            missing, unexpected = model.load_state_dict(obj, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading checkpoint: {missing[:5]}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected[:5]}")
            return model
        # Try nested state_dict
        if "state_dict" in obj:
            model.load_state_dict(obj["state_dict"], strict=False)
            return model
    # Full model object saved
    if hasattr(obj, "state_dict") and hasattr(obj, "parameters"):
        return obj
    raise ValueError(f"Unrecognised checkpoint format at {path}")


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------

def run_baseline_comparisons(in_path: Path, out_dir: Path, exp_name: str):
    import lzma, zlib

    with open(in_path, "rb") as f:
        data = f.read()
    orig_size = len(data)
    results = {}

    # LZMA
    try:
        t0 = time.perf_counter()
        comp = lzma.compress(data, preset=9)
        results["lzma"] = {
            "size": len(comp),
            "time_s": time.perf_counter() - t0,
            "path": str(out_dir / f"{exp_name}.lzma"),
        }
        with open(results["lzma"]["path"], "wb") as f:
            f.write(comp)
    except Exception as e:
        results["lzma"] = {"error": str(e)}

    # ZLIB
    try:
        t0 = time.perf_counter()
        comp = zlib.compress(data, level=9)
        results["zlib"] = {
            "size": len(comp),
            "time_s": time.perf_counter() - t0,
            "path": str(out_dir / f"{exp_name}.zlib"),
        }
        with open(results["zlib"]["path"], "wb") as f:
            f.write(comp)
    except Exception as e:
        results["zlib"] = {"error": str(e)}

    print(f"\nBaseline compression results (original: {orig_size} bytes):")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name.upper()}: ERROR — {r['error']}")
        else:
            ratio = orig_size / r["size"] if r["size"] > 0 else float("inf")
            print(
                f"  {name.upper():5} → size={r['size']} bytes | "
                f"ratio={ratio:.2f} | time={r['time_s']:.3f}s"
            )
    return results


# ---------------------------------------------------------------------------
# Interactive new-experiment creator
# ---------------------------------------------------------------------------

def create_new_experiment() -> Path:
    def _prompt(prompt, default=None, cast=str):
        suffix = f" [{default}]" if default is not None else ""
        resp = input(f"{prompt}{suffix}: ").strip()
        if resp == "" and default is not None:
            resp = str(default)
        try:
            return cast(resp)
        except Exception:
            return resp

    print("Creating a new experiment config. Press Enter to accept defaults.")
    name        = _prompt("Experiment name", "my_experiment")
    file_path   = _prompt("Path to dataset file (binary)", "/path/to/dataset.bin")
    device      = _prompt("Device (cpu|cuda)", "cuda")
    precision   = _prompt("Precision (fp32|fp16)", "fp32")
    seq_len     = _prompt("Sequence length", 32768, int)
    batch_size  = _prompt("Batch size", 3, int)
    d_model     = _prompt("d_model", 256, int)
    num_layers  = _prompt("num_layers", 4, int)
    lr          = _prompt("Learning rate", 5e-4, float)
    epochs      = _prompt("Epochs", 10, int)
    chunks      = _prompt("Compression chunks_count", 1000, int)

    cfg = {
        "name":       name,
        "file_path":  file_path,
        "progress":   True,
        "device":     device,
        "precision":  precision,
        "dataloader": {"seq_len": seq_len, "batch_size": batch_size},
        "model":      {"d_model": d_model, "num_layers": num_layers},
        "training":   {"lr": lr, "epochs": epochs},
        "compression":{"chunks_count": chunks, "file_to_compress": ""},
        "splits":     [0.8, 0.1, 0.1],
    }

    cfg_path = Path("experiments") / name / f"{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"Config written to: {cfg_path}")
    return cfg_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    timings = {}

    # ── New experiment wizard ────────────────────────────────────────────────
    if args.new_experiment:
        args.config = str(create_new_experiment())

    if args.config is None:
        raise ValueError("Provide --config <name> or use --new-experiment")

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = resolve_config_path(args.config)
    config      = load_config(config_path)
    cfg_dir     = config_path.parent

    # ── Resolve settings (CLI overrides YAML) ───────────────────────────────
    device    = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision or config.get("precision", "fp32")
    progress  = (not args.no_progress) and config.get("progress", True)
    verify    = args.verify or bool(config.get("verify", False))

    print(f"Device: {device}")

    name       = config_path.stem
    seq_len    = config.get("dataloader", {}).get("seq_len", 32768)
    batch_size = config.get("dataloader", {}).get("batch_size", 3)
    d_model    = config.get("model", {}).get("d_model", 256)
    num_layers = config.get("model", {}).get("num_layers", 4)
    lr         = float(config.get("training", {}).get("lr", 5e-4))
    num_epochs = config.get("training", {}).get("epochs", 10)
    vocab_size = 256

    exp_dir = Path("experiments") / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve model_path ───────────────────────────────────────────────────
    # Priority: CLI --model-path > YAML model_path > nothing (train from scratch)
    # --force-train always trains regardless
    model_path = None
    if not args.force_train:
        raw = args.model_path or config.get("model_path") or config.get("model", {}).get("path")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (cfg_dir / p).resolve()
            if p.exists():
                model_path = p
            else:
                print(f"[WARN] model_path '{p}' not found — will train from scratch.")

    # ── Read data ────────────────────────────────────────────────────────────
    file_path = Path(config.get("file_path", ""))
    if not file_path.is_absolute():
        file_path = (cfg_dir / file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    t0 = time.perf_counter()
    with open(file_path, "rb") as f:
        data_bytes = f.read()
    timings["read_bytes"] = time.perf_counter() - t0
    print(f"Read {len(data_bytes)} bytes from {file_path} in {timings['read_bytes']:.2f}s")

    # ── Resolve compression target ───────────────────────────────────────────
    compress_file_cfg = config.get("compression", {}).get("file_to_compress", "")
    if compress_file_cfg:
        cfp = Path(compress_file_cfg)
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp
    else:
        compress_file_path = file_path

    # ── Baseline-only mode ───────────────────────────────────────────────────
    if args.comparison_baseline_only:
        run_baseline_comparisons(compress_file_path, exp_dir, name)
        print("\n--comparison-baseline-only complete.")
        return

    # ── Build model ──────────────────────────────────────────────────────────
    model = BoaConstrictor(
        d_model=d_model,
        num_layers=num_layers,
        vocab_size=vocab_size,
        device=device,
    )

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_b, val_b, test_b = make_splits(
        data_bytes, seq_len, batch_size,
        splits=tuple(config.get("splits", (0.8, 0.1, 0.1))),
    )
    train_loader = ByteDataloader(train_b, seq_len=seq_len, batch_size=batch_size, device=device)
    val_loader   = ByteDataloader(val_b,   seq_len=seq_len, batch_size=batch_size, device=device)
    test_loader  = ByteDataloader(test_b,  seq_len=seq_len, batch_size=batch_size, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ── Load or train ────────────────────────────────────────────────────────
    skip_training = (model_path is not None)

    if skip_training:
        print(f"Loading checkpoint: {model_path}")
        t0 = time.perf_counter()
        model = load_model_from_path(model, model_path)
        timings["load_model"] = time.perf_counter() - t0
        print(f"Model loaded in {timings['load_model']:.2f}s")
    else:
        print(f"No checkpoint found — training from scratch.")

    should_train = (
        not skip_training
        and not args.compress_only
        and not args.decompress_only
        and not args.comparison_baseline_only
    )

    if should_train:
        print(f"Training: device={device} | precision={precision} | epochs={num_epochs}")
        t0 = time.perf_counter()
        train(
            model, train_loader, val_loader, test_loader,
            optimizer, criterion,
            device=device,
            name=str(exp_dir / name),
            NUM_EPOCHS=num_epochs,
            PRECISION=precision,
            progress=progress,
            start_epoch=1,
            vocab_size=vocab_size,
        )
        timings["training"] = time.perf_counter() - t0
        print(f"Training complete in {timings['training']:.2f}s")

        # Save model_path back to YAML for next run
        trained_ckpt = exp_dir / f"{name}_final_model_{precision}.pt"
        try:
            with open(config_path, "r") as f:
                cfg_data = yaml.safe_load(f) or {}
            cfg_data["model_path"] = f"{name}_final_model_{precision}.pt"
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg_data, f)
            print(f"Saved model_path to config: {trained_ckpt.name}")
        except Exception as e:
            print(f"[WARN] Could not update config with model_path: {e}")

    # Move model to device
    model = model.to(device)

    # ── Compression ──────────────────────────────────────────────────────────
    boa = BOA(device, str(exp_dir / f"{name}.boa"), model)
    file_format = compress_file_path.suffix.lstrip(".") or "bin"

    if not args.train_only and not args.decompress_only and not args.evaluate_only:
        print("Starting compression...")
        t0 = time.perf_counter()
        boa.compress(
            data_path=str(compress_file_path),
            chunks_count=config.get("compression", {}).get("chunks_count", 1000),
            progress=progress,
        )
        boa_size      = Path(exp_dir / f"{name}.boa").stat().st_size
        original_size = compress_file_path.stat().st_size
        ratio         = original_size / boa_size if boa_size > 0 else float("inf")
        timings["compression"] = time.perf_counter() - t0
        print(f"Compression ratio: {ratio:.2f}")
        print(f"Compression complete in {timings['compression']:.2f}s")

    # ── Decompression ─────────────────────────────────────────────────────────
    if not args.train_only and not args.compress_only and not args.evaluate_only:
        print("Starting decompression...")
        t0 = time.perf_counter()
        decompressed_bytes = boa.decompress(progress=progress)

        out_path = exp_dir / f"{name}_decompressed.{file_format}"
        with open(out_path, "wb") as f:
            f.write(decompressed_bytes)
        timings["decompression"] = time.perf_counter() - t0
        print(f"Decompression complete in {timings['decompression']:.2f}s")

        if verify:
            with open(compress_file_path, "rb") as f:
                ref = f.read()
            if decompressed_bytes == ref:
                print(f"VERIFY: OK — {len(decompressed_bytes)} bytes match.")
            else:
                print("VERIFY: MISMATCH")
                if len(decompressed_bytes) != len(ref):
                    print(f"  Size: decompressed={len(decompressed_bytes)} vs original={len(ref)}")
                else:
                    for i in range(min(len(decompressed_bytes), 1_000_000)):
                        if decompressed_bytes[i] != ref[i]:
                            print(f"  First mismatch at byte {i}")
                            break

    # ── Evaluation ────────────────────────────────────────────────────────────
    run_eval = (args.evaluate or args.evaluate_only) and torch.cuda.is_available()
    if run_eval:
        from evaluator import CompressionEvaluator
        print("Starting evaluation...")
        evaluator = CompressionEvaluator(model, device=device)
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        eval_loader = ByteDataloader(test_b, seq_len=1024, batch_size=1, device=device)
        evaluator.plot_topk_accuracy(
            eval_loader, k_max=20,
            savepath=str(plots_dir / "top_k_accuracy.png"),
        )
        print("Evaluation complete.")
    elif (args.evaluate or args.evaluate_only) and not torch.cuda.is_available():
        print("[WARN] Evaluation requires CUDA — skipping.")

    # ── Timings ───────────────────────────────────────────────────────────────
    if args.show_timings:
        print("\nTimings:")
        for k, v in timings.items():
            print(f"  {k:20s}: {v:.2f}s")


if __name__ == "__main__":
    main()