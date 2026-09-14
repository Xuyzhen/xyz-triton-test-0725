"""Digest-tier compare for text-only transfer channels (no scp/git-push).

When GPU and NPU servers can only exchange text (copy-paste), full .pt
tensors cannot travel. This tool works in two modes:

  1. Per side, produce a compact digest table (plain text, a few KB):
       python precision/digest_report.py --side gpu   # -> results/gpu/digest_table.txt
       python precision/digest_report.py --side npu   # -> results/npu/digest_table.txt
     Copy-paste each table to the comparison machine.

  2. Compare two tables:
       python precision/digest_report.py --compare results/gpu/digest_table.txt results/npu/digest_table.txt

Semantics:
  - IN:<name>  input digest. MISMATCH => the two sides did NOT run on
    bit-identical inputs (inputs/ are regenerated from seed per side;
    equality here is the design invariant holding).
  - OUT:<name> output digest (sha256 of contiguous CPU bytes, dtype-aware,
    device-independent). MATCH => bitwise-identical result: a hard pass
    stronger than any tolerance-based compare.
  - Outputs with mode "skip" (stochastic, e.g. sampled token ids) are
    reported SKIP: differing RNG streams are expected.
  - Outputs whose case declares a normalize hook (gumbel logits_cache:
    GPU pre-temperature vs NPU post-temperature) cannot be judged on raw
    digests -> reported NORM, needs tensor-tier compare for that case only.

Digest tables are also self-verifying: run the same side twice and diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREC_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = PREC_ROOT.parent

# kernels with normalize hooks: raw digest verdict would be a false DIFF
_NORM_KERNELS = {"gumbel_sample"}


def build_table(side: str, results_root: Path) -> str:
    import time
    cases_dir = results_root / side / "cases"
    if not cases_dir.exists():
        sys.exit(f"ERROR: no captured results under {cases_dir}; run --side {side} capture first")
    lines = [f"# digest table side={side} generated={time.strftime('%Y-%m-%d %H:%M:%S')}"]
    for meta_path in sorted(cases_dir.rglob("*.json")):
        meta = json.loads(meta_path.read_text())
        kernel, cid, case = meta["kernel"], meta["case_id"], meta["case"]
        for name, info in meta.get("inputs", {}).items():
            lines.append(f"{kernel}|{cid}|{case}|IN:{name}|-|{info['digest']}")
        modes = meta.get("output_modes", {})
        for name, info in meta.get("outputs", {}).items():
            mode = modes.get(name, "?")
            tag = "NORM" if kernel in _NORM_KERNELS else mode
            lines.append(f"{kernel}|{cid}|{case}|OUT:{name}|{tag}|{info['digest']}")
    return "\n".join(lines) + "\n"


def parse_table(text: str) -> dict[tuple[str, str, str], tuple[str, str]]:
    table: dict[tuple[str, str, str], tuple[str, str]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 6:
            continue
        kernel, cid, _case, key, tag, digest = parts
        table[(kernel, cid, key)] = (tag, digest)
    return table


def compare_tables(gpu_text: str, npu_text: str) -> int:
    gpu = parse_table(gpu_text)
    npu = parse_table(npu_text)
    rows = []
    for key in sorted(set(gpu) | set(npu)):
        g, n = gpu.get(key), npu.get(key)
        kernel, cid, field = key
        if g is None or n is None:
            rows.append((kernel, cid, field, "-", "MISSING", "captured on one side only"))
            continue
        tag, g_d = g
        _tag, n_d = n
        if field.startswith("IN:"):
            verdict = "MATCH" if g_d == n_d else "MISMATCH"
            detail = "" if g_d == n_d else "inputs differ: seed regeneration invariant broken"
        else:  # OUT:
            if tag == "skip":
                verdict, detail = "SKIP", "stochastic output, not compared"
            elif tag == "NORM":
                verdict, detail = "NORM", "normalize hook: run tensor-tier compare for this case"
            else:
                verdict = "MATCH" if g_d == n_d else "DIFF"
                detail = "" if g_d == n_d else "digests differ: run tensor-tier compare for this case"
        rows.append((kernel, cid, field, tag, verdict, detail))

    print(f"{'kernel':<26} {'case_id':<14} {'field':<28} {'mode':<8} verdict  detail")
    order = {"MISMATCH": 0, "MISSING": 1, "DIFF": 2, "NORM": 3, "SKIP": 4, "MATCH": 5}
    for kernel, cid, field, tag, verdict, detail in sorted(rows, key=lambda r: (order.get(r[4], 9), r[0], r[1], r[2])):
        print(f"{kernel:<26} {cid:<14} {field:<28} {tag:<8} {verdict:<8} {detail}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r[4]] = counts.get(r[4], 0) + 1
    print("\nsummary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    hard = counts.get("MISMATCH", 0) + counts.get("MISSING", 0) + counts.get("DIFF", 0)
    print(f"hard failures (MISMATCH/MISSING/DIFF): {hard}")
    if counts.get("MATCH"):
        print(f"bitwise-identical outputs: {counts['MATCH']}")
    return 1 if hard else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--side", choices=["gpu", "npu"])
    group.add_argument("--compare", nargs=2, metavar=("GPU_TABLE", "NPU_TABLE"))
    args = parser.parse_args()

    if args.side:
        import time
        results_root = SUITE_ROOT / "results"
        table = build_table(args.side, results_root)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = results_root / args.side / f"digest_table_{args.side}_{stamp}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(table)
        print(f"written: {out} ({len(table.splitlines())} lines) - copy-paste this file's content")
        return 0

    gpu_text = Path(args.compare[0]).read_text(encoding="utf-8")
    npu_text = Path(args.compare[1]).read_text(encoding="utf-8")
    return compare_tables(gpu_text, npu_text)


if __name__ == "__main__":
    raise SystemExit(main())
