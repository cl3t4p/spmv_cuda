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
xmake run spmv <dtype> <matrix_type> <file.mtx>
```

| Argument | Values |
|---|---|
| `dtype` | `int`, `float` |
| `matrix_type` | `coo`, `csr_scalar`, `csr_vec`, `ell` |
| `file.mtx` | Matrix Market file |

Example:

```
xmake run spmv float csr_vec data/matrix.mtx
```

Each run performs 10 warmup iterations followed by 100 timed iterations, then prints the CPU time, average GPU time, and the error between the two results.

## Project layout

- `src/` — CUDA kernels and matrix loader
- `inc/` — headers
- `data/` — input matrices (Matrix Market format)
- `report/` — report sources
- `sbatch.sh` — Slurm submission script
