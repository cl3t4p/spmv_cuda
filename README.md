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

## Build

Built with [xmake](https://xmake.io):

```
xmake
```

The build targets `sm_80` (Ampere), links against cuSPARSE, and uses OpenMP for the CPU reference.

## Usage

```
xmake run spmv <dtype> <matrix_type> <file.mtx> [options]
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
xmake run spmv float csr_vec data/matrix.mtx
```

Each run performs the configured warmup and timed iterations, then prints the CPU time and GFlops, the GPU mean time and GFlops, arithmetic/geometric mean and standard deviation across runs, and the error between CPU and GPU results. With `--conversion`, the COO → target-format host conversion time is reported as well.

## Project layout

- `src/` — CUDA kernels and matrix loader
- `inc/` — headers
- `data/` — input matrices (Matrix Market format)
- `report/` — report sources
- `sbatch.sh` — Slurm submission script
