# BoaConstrictor: Neural Lossless Compressor for High Energy Physics Data [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> **GSoC 2025 Warm-Up Fork** — This fork replaces the Mamba backbone with MinGRU, achieving **4.18× compression ratio** on CMS Open Data. See [Warm-Up Task Results](#gsoc-2025-warm-up-task-results) below.

---

This repo provides a byte-level compression pipeline driven by a neural predictor (BoaConstrictor) and entropy coding (range coding). It includes:

- A clean CLI to train a model, compress with it, and decompress back
- Per-experiment YAML configs and an interactive config creator
- Optional progress bars and timing for each major stage
- CPU and GPU execution, with tips for best performance

Key entrypoints:
- CLI: `main.py`
- Example config: `experiments/cms_experiment/cms_experiment.yaml`

> [!NOTE]
> **Reference implementation for GPU Portability**
> The `portability_solved_cpp` folder contains a reference implementation of BOA using the Mamba network in C++. This implementation specifically solves portability issues on GPUs for CUDA. Please note that it includes only compression/decompression logic and does not contain code for training.

---

## GSoC 2025 Warm-Up Task Results

### What was changed

The original BOA backbone uses **Mamba** (a state-space model requiring `mamba-ssm` + `causal-conv1d`, both CUDA-only libraries). These fail to build on standard Colab T4 runtimes. This fork replaces Mamba with **MinGRU** — a minimal gated recurrent unit implemented entirely in PyTorch with zero additional dependencies.

**New files:**
- `model_mingru.py` — MinGRU backbone (drop-in replacement for `model.py`)
- `main.py` — rewritten CLI (explicit model path, no auto-skip on checkpoint detection)
- `experiments/cms_experiment/baseline_results.yaml` — full benchmark results

### Benchmark Results — CMS Open Data (49.9 MB float32)

| Method | Type | Compression Ratio |
|--------|------|:-----------------:|
| **MinGRU (30 epochs)** | **Neural** | **4.18×** |
| **MinGRU (5 epochs)** | **Neural** | **3.89×** |
| LZMA (preset=9) | Traditional | 3.26× |
| Brotli (quality=11) | Traditional | 3.05× |
| ByteShuffle + LZMA | Physics-Aware | 3.05× |
| ZSTD (level=22) | Traditional | 2.92× |
| BZ2 (level=9) | Traditional | 2.75× |
| Blosc2 + SHUFFLE (float32) | Scientific Array | 2.57× |
| ZLIB (level=9) | Traditional | 2.56× |
| Blosc2 (ZSTD, clevel=9) | Scientific Array | 2.52× |
| Delta + LZMA | Physics-Aware | 2.40× |
| LZ4 (max) | Traditional | 2.33× |

**Key numbers:**
- MinGRU beats LZMA (best traditional) by **+28%**
- MinGRU beats Blosc2 (HEP standard) by **+66%**
- ByteShuffle+LZMA uses explicit float32 domain knowledge — MinGRU matches and surpasses it automatically

### Training curve

| Epoch | Val bpp | Ratio |
|-------|---------|-------|
| 1 | 2.1896 | 3.65× |
| 8 | 1.9985 | 4.00× |
| 21 | 1.9246 | 4.16× |
| 30 | 1.9153 | **4.18×** |

Model converges cleanly. No overfitting observed through 30 epochs.

### Timings (T4 GPU, fp32)

| Stage | Duration |
|-------|----------|
| Training (30 epochs) | 143 min |
| Compression (49.9 MB) | 43s |
| Decompression | 43s |
| Verification | ✅ PASS — 49,920,000 bytes match |

### Architecture

Both Mamba and MinGRU use the same interface:

```
Embedding(256 → d_model) → [Block × num_layers] → Linear(d_model → 256) → Softmax → Range Coder
```

Config used: `d_model=256`, `num_layers=4`, `vocab_size=256`, `epochs=30`, `lr=5e-4`, `precision=fp32`

### Why MinGRU over Mamba

| | Mamba | MinGRU |
|---|---|---|
| Dependencies | mamba-ssm + causal-conv1d (CUDA-only) | Pure PyTorch |
| Colab T4 | Build errors | Works out of the box |
| Inference | O(1) per step via parallel scan | O(1) per step via hidden state |
| Portability | GPU-locked | CPU, GPU, HPC, edge |

### Key bug fixed

The GRU defaults to eval mode inside PyTorch's `inference_mode` context, causing a `cudnn RNN backward can only be called in training mode` error during training. Fix applied in `model_mingru.py`:

```python
def forward(self, x, inference_params=None):
    if torch.is_grad_enabled():
        self.gru.train()
    out, _ = self.gru(x)
    return out
```

---

## Quick Start (MinGRU)

```bash
# 1. Clone and install
git clone https://github.com/Shubhf/boa-constrictor.git
cd boa-constrictor
pip install pybind11 ninja numpy PyYAML tqdm

# 2. Add your dataset
# Place CMS_DATA_float32.bin in experiments/cms_experiment/

# 3. Train MinGRU from scratch
python main.py --config cms_experiment --show-timings --verify

# 4. Run all baselines for comparison
python main_baseline.py --config cms_experiment --all --compare
```

To restore from a checkpoint:
```python
# In your YAML, set:
model_path: experiments/cms_experiment/cms_experiment_final_model_fp32.pt
```

---

## Quick start (original)

1) Install dependencies (PyTorch not pinned here; use the build suited for your system):

```bash
python3 -m pip install -r requirements.txt
```

2) Create a config interactively and run the experiment:

```bash
python3 main.py --new-experiment
```

3) Or run with an existing config and show timings:

```bash
python3 main.py --config experiment_name --show-timings
```

Useful flags:
- `--no-progress` to disable progress bars
- `--device cpu|cuda` to override device
- `--precision fp32|fp16|fp8` to override compute precision (training only)
- `--train-only`, `--compress-only`, `--decompress-only` to run specific stages
- `--model-path /path/to/model.pt` to load a pre-trained checkpoint and skip training
- `--verify` to verify the files after compression-decompression cycle
- `--evaluate`, `--evaluate-only` to evaluate performance of the compression model
- `--comparison-baseline-only` to run LZMA and ZLIB on the dataset as baselines

> [!WARNING]
> Currently training can only be done on a CUDA-Compatible GPU!

---

## Config file structure

A minimal example (`configs/experiment.yaml`):

```yaml
name: example_experiment
file_path: /path/to/dataset.bin
progress: true
device: cuda
precision: fp16

# Optional: set a checkpoint to skip training
# model_path: /path/to/checkpoints/example_experiment_final_model_fp16.pt

dataloader:
  seq_len: 32768
  batch_size: 3

model:
  d_model: 256
  num_layers: 8

training:
  lr: 5e-4
  epochs: 50

compression:
  chunks_count: 1000
  file_to_compress: ''

splits: [0.8, 0.1, 0.1]
```

---

## Architecture and data flow

1) **Byte modeling (neural predictor)**
   - The `BoaConstrictor` model receives byte sequences and predicts a distribution over the next byte (0..255) at each position.
   - Training minimizes cross-entropy between predictions and observed bytes.

2) **Entropy coding (range coding)**
   - For each byte to be stored, the predictor provides probabilities p(b | context).
   - A range coder converts these probabilities and symbols into a compact bitstream close to the theoretical entropy (−log₂ p).

3) **Container and chunks**
   - Data is processed in chunks, enabling parallelism and streaming.
   - Each chunk stores (a) first bytes, (b) the compressed range-coded stream, and (c) metadata.

4) **Decompression mirrors compression**
   - The range decoder reconstructs each symbol using the same probabilities generated by the model conditioned on previously decoded bytes.

---

## Range coding primer

Range coding is a practical form of arithmetic coding. It maintains an interval [low, high) within [0, 1). For each symbol with probability distribution {p_i}:

- Partition the current interval into sub-intervals proportional to {p_i}
- Select the sub-interval for the observed symbol
- Renormalize when the interval becomes too small, emitting bits

Total code length approaches −Σ_t log₂ p(x_t | context) bits.

```
state: low=0, high=RANGE_MAX
for symbol s with cumulative counts C and total T:
  range = high - low + 1
  high  = low + (range * C[s+1] // T) - 1
  low   = low + (range * C[s]   // T)
  while renormalization_condition(low, high):
    output_bit_and_shift(low, high)
```

---

## Performance notes

### Streaming compression batches

```bash
export BOA_GPU_STREAMS=5000
python3 main.py --config your_experiment
```

### CPU speedups
- Vectorized preprocessing with NumPy/PyTorch tensor operations
- Tune `chunks_count` to fit chunks in cache
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads` for CPU threading

### GPU speedups
- Batch inference across longer sequences
- Mixed precision: `--precision fp16`
- Chunk-level parallelism to keep GPU fed

---

## Troubleshooting

- `file_path` not found — update YAML to point to existing dataset
- CUDA out of memory — reduce `batch_size` or `seq_len`
- Slow GPU throughput — increase chunk parallelism, avoid tiny chunks
- `mamba-ssm` build errors — use MinGRU backbone (`model_mingru.py`), no CUDA libraries needed

---

## References

- BOA Constrictor paper: [arXiv:2511.11337](https://arxiv.org/abs/2511.11337)
- Range coding: classic arithmetic coding literature
- MinGRU: [Were RNNs All We Needed? (2024)](https://arxiv.org/abs/2410.01201)
- CMS Open Data: [opendata.cern.ch](https://opendata.cern.ch)

## Citation

```bibtex
@misc{gupta2025boaconstrictormambabasedlossless,
      title={BOA Constrictor: A Mamba-based lossless compressor for High Energy Physics data},
      author={Akshat Gupta and Caterina Doglioni and Thomas Joseph Elliott},
      year={2025},
      eprint={2511.11337},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2511.11337},
}
```

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. See [LICENSE](LICENSE) for details.
