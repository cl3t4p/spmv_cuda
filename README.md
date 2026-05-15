# spmv_cuda

A small CUDA project exploring **Sparse Matrix–Vector multiplication (SpMV)** across different sparse storage formats. Built for a university GPU course, it benchmarks each format against a CPU baseline and reports timing and numerical error.

## Supported formats

- **COO** (`coo`) — coordinate list
- **COO optimized** (`coo_opt`) — coordinate list with a tuned reduction
- **CSR scalar** (`csr_scalar`) — one thread per row
- **CSR vector** (`csr_vec`) — one warp per row
- **ELL** (`ell`) — fixed-width row storage
- **cuSPARSE CSR** (`csr_cusparse`) — vendor reference (float/double only)
- **cuSPARSE COO** (`coo_cusparse`) — vendor reference (float/double only)

## Prerequisites

Fetch the SuiteSparse matrices before building or running anything:

```
./download_data.sh
```

This populates `data/` with the 10 matrices used in the deliverable (existing ones are skipped).

## Build

Built with [xmake](https://xmake.io). A `xmakew` wrapper (gradlew-style) is included — it uses the system `xmake` if available, otherwise downloads and builds a pinned release locally:

```
./xmakew
```

## Local Usage

```
./xmakew run spmv <dtype> <matrix_type> <file.mtx> [options]
```

| Argument | Values |
|---|---|
| `dtype` | `int`, `float` |
| `matrix_type` | `coo`, `coo_opt`, `csr_scalar`, `csr_vec`, `ell`, `csr_cusparse`, `coo_cusparse` |
| `file.mtx` | Matrix Market file |

Optional flags:

| Flag | Description |
|---|---|
| `--seed <int>` | seed for the dense-vector RNG (default: random) |
| `--conversion` | also time the COO → `matrix_type` host conversion |
| `--gpu-warmup <int>` | GPU warmup launches (default 10) |
| `--gpu-runs <int>` | GPU timed launches (default 100) |
| `--conv-warmup <int>` | conversion warmup runs (default 2) |
| `--conv-runs <int>` | conversion timed runs (default 5) |

Example:

```
./xmakew run spmv float csr_vec data/matrix.mtx
```

Each run performs the configured warmup and timed iterations, then prints the CPU time and GFlops, the GPU mean time and GFlops, arithmetic/geometric mean and standard deviation across runs, and the error between CPU and GPU results. With `--conversion`, the COO → target-format host conversion time is reported as well.

## Batch runs with `run_all.py`

`run_all.py` submits one Slurm job per `(matrix, format)` pair sequentially, waiting for each to finish before submitting the next. It scans `data/` for `.mtx` files (both flat and `name/name.mtx` layouts) and submits via the sbatch script you pass as the first argument.

```
./run_all.py sbatch_l40s.sh          # all formats on L40S
./run_all.py sbatch_a30.sh           # all formats on A30
```

Options:

| Flag | Description |
|---|---|
| `--data <dir>` | data folder (default: `data`) |
| `--dtype {int,float,auto}` | value type; `auto` picks `int` for pattern/integer matrices, `float` otherwise (default: `auto`) |
| `--formats <fmt...>` | subset of formats to run (default: all) |
| `--no-conversion` | don't pass `--conversion` to `spmv` (default: enabled) |
| `--seed <int>` | RNG seed passed to `spmv` (default: 42) |

cuSPARSE formats are skipped automatically when `dtype` resolves to `int`. Ctrl-C cancels the currently running job via `scancel`.

Example — only the CSR kernels on L40S with a fixed seed:

```
./run_all.py sbatch_l40s.sh --formats csr_scalar csr_vec --seed 42
```

## Project layout

- `src/` — CUDA kernels and matrix loader
- `inc/` — headers
- `data/` — input matrices (Matrix Market format), populated by `download_data.sh`
- `report/` — report sources
- `sbatch_l40s.sh`, `sbatch_a30.sh` — Slurm submission scripts
- `run_all.py` — sweep driver that submits one job per `(matrix, format)`
- `xmakew` — xmake wrapper (downloads a pinned xmake if none is installed)
