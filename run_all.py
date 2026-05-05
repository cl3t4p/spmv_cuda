#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path

FORMATS = ["coo", "coo_opt", "csr_scalar", "csr_vec", "ell", "csr_cusparse", "coo_cusparse"]
CUSPARSE_FORMATS = {"csr_cusparse", "coo_cusparse"}
DTYPES = ["int", "float", "auto"]
INT_FIELDS = {"pattern", "integer"}


def find_matrices(data_dir: Path):
    matrices = []
    for entry in sorted(data_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".mtx":
            matrices.append(entry)
        elif entry.is_dir():
            candidate = entry / f"{entry.name}.mtx"
            if candidate.is_file():
                matrices.append(candidate)
    return matrices


def detect_dtype(mtx_path: Path) -> str:
    with mtx_path.open() as f:
        header = f.readline().strip().lower()
    parts = header.split()
    if len(parts) < 5 or not parts[0].startswith("%%matrixmarket"):
        return "float"
    field = parts[3]
    return "int" if field in INT_FIELDS else "float"


def submit(sbatch_script: Path, spmv_args: list[str]) -> str:
    cmd = ["sbatch", "--parsable", str(sbatch_script), *spmv_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip().split(";")[0]


def job_active(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def wait_for(job_id: str, poll: int) -> None:
    while job_active(job_id):
        time.sleep(poll)


def main() -> int:
    p = argparse.ArgumentParser(description="Run spmv sbatch jobs sequentially over all matrices and formats.")
    p.add_argument("--data", type=Path, default=Path("data"), help="data folder (default: data)")
    p.add_argument("--sbatch", type=Path, default=Path("sbatch.sh"), help="sbatch script (default: sbatch.sh)")
    p.add_argument("--dtype", default="auto", choices=DTYPES, help="value type; auto picks int for pattern/integer matrices, float otherwise (default: auto)")
    p.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS, help="formats to run (default: all)")
    p.add_argument("--conversion", action=argparse.BooleanOptionalAction, default=True, help="pass --conversion to spmv (default: enabled; use --no-conversion to disable)")
    p.add_argument("--poll", type=int, default=10, help="seconds between squeue polls (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="dense-vector RNG seed passed to spmv (default: 42)")
    p.add_argument("--dry-run", action="store_true", help="print commands without submitting")
    args = p.parse_args()

    if not args.sbatch.is_file():
        print(f"sbatch script not found: {args.sbatch}", file=sys.stderr)
        return 1
    if not args.data.is_dir():
        print(f"data folder not found: {args.data}", file=sys.stderr)
        return 1

    matrices = find_matrices(args.data)
    if not matrices:
        print(f"no .mtx files found in {args.data}", file=sys.stderr)
        return 1

    total = len(matrices) * len(args.formats)
    i = 0
    for m in matrices:
        dtype = detect_dtype(m) if args.dtype == "auto" else args.dtype
        for fmt in args.formats:
            i += 1
            if dtype == "int" and fmt in CUSPARSE_FORMATS:
                print(f"[{i}/{total}] skipping {fmt} for {m.name} (cuSPARSE requires float/double)", flush=True)
                continue
            spmv_args = [dtype, fmt, str(m), "--seed", str(args.seed)]
            if args.conversion:
                spmv_args.append("--conversion")

            print(f"[{i}/{total}] {' '.join(spmv_args)}", flush=True)
            if args.dry_run:
                continue

            try:
                job_id = submit(args.sbatch, spmv_args)
            except subprocess.CalledProcessError as e:
                print(f"  sbatch failed: {e.stderr.strip()}", file=sys.stderr)
                return 1

            print(f"  job {job_id} submitted", flush=True)
            try:
                wait_for(job_id, args.poll)
            except KeyboardInterrupt:
                print(f"\ninterrupted; cancelling job {job_id}", file=sys.stderr)
                subprocess.run(["scancel", job_id])
                return 130
            print(f"  job {job_id} done", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
