"""
main_baseline.py — Comprehensive Baseline Benchmarking for BoaConstrictor

Compressors included:
  GENERAL PURPOSE (built-in):
    - ZLIB      : Python standard, deflate algorithm
    - BZ2       : Python standard, block-sort compressor
    - LZMA      : Python standard, strongest traditional compressor

  GENERAL PURPOSE (pip install):
    - LZ4       : Extremely fast, lower ratio — pip install lz4
    - Zstandard : Facebook's modern compressor — pip install zstandard
    - Brotli    : Google's compressor — pip install brotli

  SCIENTIFIC / NUMERICAL (pip install):
    - Blosc2            : Designed for numerical arrays (HDF5/NumPy) — pip install blosc2
    - Blosc2 + SHUFFLE  : Float32-aware byte shuffle filter (best for CMS data)

  PHYSICS-AWARE (no extra deps):
    - Delta + LZMA      : Delta-encode float32 stream, then LZMA
    - ByteShuffle + LZMA: Regroup bytes by float32 significance, then LZMA

Usage:
  python main_baseline.py --config cms_experiment --all --compare
  python main_baseline.py --config cms_experiment --traditional-only --compare
  python main_baseline.py --config cms_experiment --scientific-only --compare

  # After MinGRU 30-epoch run completes:
  python main_baseline.py --config cms_experiment --compare --update-mingru-30 4.8
"""

import argparse
import bz2
import lzma
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Optional dependency loader
# ---------------------------------------------------------------------------

def try_import(package: str, pip_name: str = None):
    """Try importing package; auto-install via pip if missing."""
    try:
        return __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"    [auto-install] pip install {pip_name} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            capture_output=True,
        )
        if result.returncode == 0:
            try:
                return __import__(package)
            except ImportError:
                pass
        print(f"    [SKIP] {package} could not be installed.")
        return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_arg: str) -> Path:
    p = Path(config_arg)
    if p.exists():
        return p
    exp = Path("experiments") / p.stem / f"{p.stem}.yaml"
    if exp.exists():
        return exp
    raise FileNotFoundError(f"Cannot resolve config: {config_arg}")


# ---------------------------------------------------------------------------
# Compressor runners
# Each returns: {ratio, compressed_size, original_size, time_s, level}
#           or: {error: str}
# ---------------------------------------------------------------------------

def run_zlib(data: bytes, level: int = 9) -> dict:
    t0   = time.perf_counter()
    comp = zlib.compress(data, level=level)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": f"level={level}",
    }


def run_bz2(data: bytes, level: int = 9) -> dict:
    t0   = time.perf_counter()
    comp = bz2.compress(data, compresslevel=level)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": f"level={level}",
    }


def run_lzma(data: bytes, preset: int = 9) -> dict:
    t0   = time.perf_counter()
    comp = lzma.compress(data, preset=preset)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": f"preset={preset}",
    }


def run_lz4(data: bytes) -> dict:
    mod = try_import("lz4.frame", "lz4")
    if mod is None:
        return {"error": "lz4 not available"}
    import lz4.frame
    t0   = time.perf_counter()
    comp = lz4.frame.compress(data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": "max",
    }


def run_zstandard(data: bytes, level: int = 22) -> dict:
    zstd = try_import("zstandard")
    if zstd is None:
        return {"error": "zstandard not available"}
    cctx = zstd.ZstdCompressor(level=level)
    t0   = time.perf_counter()
    comp = cctx.compress(data)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": f"level={level}",
    }


def run_brotli(data: bytes, quality: int = 11) -> dict:
    brotli = try_import("brotli")
    if brotli is None:
        return {"error": "brotli not available"}
    t0   = time.perf_counter()
    comp = brotli.compress(data, quality=quality)
    return {
        "ratio": len(data) / len(comp),
        "compressed_size": len(comp),
        "original_size": len(data),
        "time_s": time.perf_counter() - t0,
        "level": f"quality={quality}",
    }


def run_blosc2_best(data: bytes) -> dict:
    """
    Blosc2: designed for numerical arrays.
    Tries BLOSCLZ, LZ4HC, ZSTD internally. Reports best ratio.
    No shuffle filter — raw byte stream compression.
    """
    blosc2 = try_import("blosc2")
    if blosc2 is None:
        return {"error": "blosc2 not available"}

    best = {}
    codec_map = {
        "blosclz": blosc2.Codec.BLOSCLZ,
        "lz4hc":   blosc2.Codec.LZ4HC,
        "zstd":    blosc2.Codec.ZSTD,
    }
    for codec_name, codec in codec_map.items():
        try:
            t0   = time.perf_counter()
            comp = blosc2.compress(data, codec=codec, clevel=9, nthreads=1)
            ratio = len(data) / len(comp)
            if ratio > best.get("ratio", 0):
                best = {
                    "ratio": ratio,
                    "compressed_size": len(comp),
                    "original_size": len(data),
                    "time_s": time.perf_counter() - t0,
                    "level": f"codec={codec_name},clevel=9",
                }
        except Exception:
            pass
    return best if best else {"error": "blosc2 compression failed"}


def run_blosc2_shuffle(data: bytes) -> dict:
    """
    Blosc2 + BYTE SHUFFLE filter, typesize=4 (float32-aware).

    How SHUFFLE helps CMS data:
      Original memory layout per 3 floats (12 bytes):
        [S E E M M M | S E E M M M | S E E M M M]
        (interleaved sign+exponent+mantissa bytes)

      After SHUFFLE (bytes reordered by position within float):
        [S S S | E E E | M M M | M M M | M M M | M M M]
        (all sign bytes together, all exponent bytes together, etc.)

      Consecutive exponent bytes in CMS physics data are nearly identical
      (measurements in same range) → very high compressibility.
      This is what MinGRU learns to exploit automatically via context.
    """
    blosc2 = try_import("blosc2")
    if blosc2 is None:
        return {"error": "blosc2 not available"}

    try:
        t0   = time.perf_counter()
        comp = blosc2.compress(
            data,
            codec=blosc2.Codec.ZSTD,
            clevel=9,
            filter=blosc2.Filter.SHUFFLE,
            typesize=4,   # float32 = 4 bytes
            nthreads=1,
        )
        return {
            "ratio": len(data) / len(comp),
            "compressed_size": len(comp),
            "original_size": len(data),
            "time_s": time.perf_counter() - t0,
            "level": "zstd+SHUFFLE(float32)",
        }
    except Exception as e:
        return {"error": str(e)}


def run_delta_lzma(data: bytes) -> dict:
    """
    Delta encoding + LZMA — manual physics-aware baseline.

    Idea: consecutive CMS float32 values are similar (sequential measurements).
    Store differences (deltas) instead of absolute values.
    Deltas are small numbers → more compressible than raw floats.

    This is a hand-crafted version of what a sequence model learns automatically.
    Comparing this vs MinGRU shows how much ML adds beyond simple delta coding.
    """
    try:
        arr    = np.frombuffer(data, dtype=np.float32).copy()
        deltas = np.diff(arr, prepend=arr[0]).astype(np.float32).tobytes()
        t0     = time.perf_counter()
        comp   = lzma.compress(deltas, preset=9)
        return {
            "ratio": len(data) / len(comp),
            "compressed_size": len(comp),
            "original_size": len(data),
            "time_s": time.perf_counter() - t0,
            "level": "float32_delta+lzma9",
        }
    except Exception as e:
        return {"error": str(e)}


def run_byteshuffle_lzma(data: bytes) -> dict:
    """
    Manual byte-shuffle + LZMA — float32-aware without Blosc2.

    Regroups bytes by their role within float32:
      byte 0 = sign + upper exponent bits  (very similar across CMS measurements)
      byte 1 = lower exponent bits          (similar)
      byte 2 = upper mantissa bits          (variable)
      byte 3 = lower mantissa bits          (most variable)

    After reordering, LZMA sees runs of similar bytes → better compression.
    Same idea as Blosc2 SHUFFLE but without the library dependency.
    """
    try:
        n      = len(data) // 4
        arr    = np.frombuffer(data[:n * 4], dtype=np.uint8).reshape(n, 4)
        shuffled = arr.T.tobytes()   # shape (4, n) → group all byte-0s, all byte-1s, etc.
        t0     = time.perf_counter()
        comp   = lzma.compress(shuffled, preset=9)
        return {
            "ratio": len(data) / len(comp),
            "compressed_size": len(comp),
            "original_size": len(data),
            "time_s": time.perf_counter() - t0,
            "level": "byteshuffle+lzma9",
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

CATEGORIES = {
    "zlib":                   "General Purpose",
    "bz2":                    "General Purpose",
    "lzma":                   "General Purpose",
    "lz4":                    "General Purpose",
    "zstandard":              "General Purpose",
    "brotli":                 "General Purpose",
    "blosc2_best":            "Scientific / Numerical",
    "blosc2_float32_shuffle": "Scientific / Numerical",
    "delta_lzma":             "Physics-Aware (manual)",
    "byteshuffle_lzma":       "Physics-Aware (manual)",
}

# Updated by --update-mingru-30 flag
MINGRU_REFERENCE = {
    "5_epochs":  {"ratio": 3.89, "time_s": 43.0,  "note": "MinGRU  (5 epochs)"},
    "30_epochs": {"ratio": None, "time_s": None,   "note": "MinGRU (30 epochs) — pending"},
}


def print_comparison_table(results: dict, original_size: int):
    print("\n" + "=" * 78)
    print(f"{'COMPRESSION BASELINE COMPARISON':^78}")
    print(f"{'Original: ' + f'{original_size/1e6:.1f} MB ({original_size:,} bytes)':^78}")
    print("=" * 78)
    print(f"  {'Method':<32} {'Ratio':>7}  {'Size':>8}  {'Time':>8}  Progress")
    print(f"  {'-'*70}")

    printed_cats = set()
    rows = []
    for name, r in results.items():
        cat = CATEGORIES.get(name, "Other")
        rows.append((cat, name, r))
    rows.sort(key=lambda x: (x[0], -x[2].get("ratio", 0)))

    for cat, name, r in rows:
        if cat not in printed_cats:
            print(f"\n  [{cat}]")
            printed_cats.add(cat)

        if "error" in r:
            print(f"  {'  ' + name:<32} {'ERROR':>7}  — {r['error']}")
            continue

        ratio     = r.get("ratio", 0)
        comp_mb   = r.get("compressed_size", 0) / 1e6
        t         = r.get("time_s", 0)
        bar       = "█" * int(ratio * 3)
        print(f"  {'  ' + name:<32} {ratio:>6.2f}x  {comp_mb:>6.1f}MB  {t:>7.2f}s  {bar}")

    # MinGRU reference rows
    print(f"\n  [Neural — MinGRU (BOA)]")
    for key, ref in MINGRU_REFERENCE.items():
        if ref["ratio"]:
            bar = "█" * int(ref["ratio"] * 3)
            comp_mb = original_size / ref["ratio"] / 1e6
            t_str   = f"{ref['time_s']:>7.2f}s" if ref["time_s"] else "       ?"
            print(f"  {'  ' + ref['note']:<32} {ref['ratio']:>6.2f}x  {comp_mb:>6.1f}MB  {t_str}  {bar}")
        else:
            print(f"  {'  ' + ref['note']:<32} {'pending':>7}")

    print("\n" + "=" * 78)
    print("\nNarrative for presentation:")
    print("  - Physics-aware (manual) methods show what explicit domain knowledge buys.")
    print("  - MinGRU learns this structure automatically — no manual feature engineering.")
    print("  - Gap between 'byteshuffle+lzma' and MinGRU = value of learned representations.")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive baseline benchmarks for BoaConstrictor"
    )
    p.add_argument("--config", "-c", type=str, required=True)
    p.add_argument("--all", action="store_true",
                   help="Run all baselines")
    p.add_argument("--traditional-only", action="store_true",
                   help="Run ZLIB + BZ2 + LZMA + LZ4 + ZSTD + Brotli only")
    p.add_argument("--scientific-only", action="store_true",
                   help="Run Blosc2 + physics-aware baselines only")
    p.add_argument("--compare", action="store_true",
                   help="Print formatted comparison table")
    p.add_argument("--update-mingru-30", type=float, default=None,
                   metavar="RATIO",
                   help="Set MinGRU 30-epoch ratio in comparison table")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    config_path  = resolve_config_path(args.config)
    config       = load_config(config_path)
    cfg_dir      = config_path.parent
    name         = config_path.stem
    exp_dir      = Path("experiments") / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    file_path = Path(config.get("file_path", ""))
    if not file_path.is_absolute():
        file_path = (cfg_dir / file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"\n{'='*60}")
    print(f"BOA Comprehensive Baseline Benchmarks — {name}")
    print(f"File: {file_path.name}  ({file_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}\n")

    with open(file_path, "rb") as f:
        data = f.read()
    original_size = len(data)

    if args.update_mingru_30:
        MINGRU_REFERENCE["30_epochs"]["ratio"]  = args.update_mingru_30
        MINGRU_REFERENCE["30_epochs"]["time_s"] = 0
        print(f"MinGRU 30-epoch reference set to {args.update_mingru_30}x\n")

    results  = {}
    run_trad = args.traditional_only or args.all
    run_sci  = args.scientific_only  or args.all

    # Default: run everything
    if not any([args.traditional_only, args.scientific_only, args.all]):
        run_trad = run_sci = True

    # ── General-purpose compressors ──────────────────────────────────────
    if run_trad:
        print("── General-purpose compressors ──────────────────────────────")

        print("  ZLIB  (level=9)...", end=" ", flush=True)
        results["zlib"] = run_zlib(data)
        r = results["zlib"]
        print(f"{r['ratio']:.2f}x  {r['time_s']:.2f}s")

        print("  BZ2   (level=9)...", end=" ", flush=True)
        results["bz2"] = run_bz2(data)
        r = results["bz2"]
        print(f"{r['ratio']:.2f}x  {r['time_s']:.2f}s")

        print("  LZMA  (preset=9)...", end=" ", flush=True)
        results["lzma"] = run_lzma(data)
        r = results["lzma"]
        print(f"{r['ratio']:.2f}x  {r['time_s']:.2f}s")

        print("  LZ4   (max)...")
        results["lz4"] = run_lz4(data)
        r = results["lz4"]
        print(f"  → {r['ratio']:.2f}x  {r['time_s']:.2f}s" if "error" not in r else f"  → {r['error']}")

        print("  ZSTD  (level=22)...")
        results["zstandard"] = run_zstandard(data, level=22)
        r = results["zstandard"]
        print(f"  → {r['ratio']:.2f}x  {r['time_s']:.2f}s" if "error" not in r else f"  → {r['error']}")

        print("  Brotli (quality=11)...")
        results["brotli"] = run_brotli(data, quality=11)
        r = results["brotli"]
        print(f"  → {r['ratio']:.2f}x  {r['time_s']:.2f}s" if "error" not in r else f"  → {r['error']}")

    # ── Scientific / physics-aware ────────────────────────────────────────
    if run_sci:
        print("\n── Scientific / physics-aware compressors ───────────────────")

        print("  Blosc2 (best codec, clevel=9)...")
        results["blosc2_best"] = run_blosc2_best(data)
        r = results["blosc2_best"]
        print(f"  → {r['ratio']:.2f}x  [{r.get('level','')}]" if "error" not in r else f"  → {r['error']}")

        print("  Blosc2 + SHUFFLE filter (float32-aware)...")
        results["blosc2_float32_shuffle"] = run_blosc2_shuffle(data)
        r = results["blosc2_float32_shuffle"]
        print(f"  → {r['ratio']:.2f}x  [{r.get('level','')}]" if "error" not in r else f"  → {r['error']}")

        print("  Delta encoding + LZMA...", end=" ", flush=True)
        results["delta_lzma"] = run_delta_lzma(data)
        r = results["delta_lzma"]
        print(f"{r['ratio']:.2f}x  {r['time_s']:.2f}s" if "error" not in r else r["error"])

        print("  Byte-shuffle + LZMA (manual float32)...", end=" ", flush=True)
        results["byteshuffle_lzma"] = run_byteshuffle_lzma(data)
        r = results["byteshuffle_lzma"]
        print(f"{r['ratio']:.2f}x  {r['time_s']:.2f}s" if "error" not in r else r["error"])

    # ── Comparison table ──────────────────────────────────────────────────
    if args.compare:
        print_comparison_table(results, original_size)

    # ── Save to YAML ──────────────────────────────────────────────────────
    out_yaml = exp_dir / "baseline_results.yaml"
    safe = {
        k: {kk: (float(vv) if hasattr(vv, "item") else vv) for kk, vv in v.items()}
        for k, v in results.items()
    }
    with open(out_yaml, "w") as f:
        yaml.safe_dump(safe, f)
    print(f"Results saved → {out_yaml}")

    print("\nAfter MinGRU 30-epoch run, add its ratio to the table:")
    print("  python main_baseline.py --config cms_experiment --compare --update-mingru-30 <ratio>\n")


if __name__ == "__main__":
    main()