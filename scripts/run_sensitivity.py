#!/usr/bin/env python3
"""Run simple parameter sensitivity experiments for RC-FAD."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--param", choices=["epsilon-fpr", "c-fn", "beta-max", "beta-alpha", "fpr-penalty"], default="epsilon-fpr")
    parser.add_argument("--values", nargs="+", default=None)
    args = parser.parse_args()

    defaults = {
        "epsilon-fpr": ["0.01", "0.03", "0.05", "0.10"],
        "c-fn": ["1.0", "2.0", "3.0", "5.0"],
        "beta-max": ["3.0", "5.0", "8.0", "12.0"],
        "beta-alpha": ["0.0", "0.3", "0.6", "1.0"],
        "fpr-penalty": ["0.5", "1.0", "2.0"],
    }
    values = args.values or defaults[args.param]
    for v in values:
        extra = f"{args.param}={v}"
        cmd = [
            sys.executable,
            "scripts/run_experiments.py",
            "--suite", "quick",
            "--methods", "rcfad",
            "--datasets", args.dataset,
            "--seeds", str(args.seed),
            "--rounds", str(args.rounds),
            "--extra", extra,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
