"""
Grid sweep over the two knobs that matter for bounded-ascent unlearning:
    --lambda-forget   how hard we push L_forget up
    --forget-margin   where the bounded-ascent term saturates

For each (lambda_forget, forget_margin) pair it runs unlearn_logic_2.py,
tags the output dir, parses the evaluation file, and at the end prints a
ranked table + best config. The "score" rewards forgetting_efficacy > 1
while penalising fidelity/generalization drift away from 1.0 — i.e., it
looks for the best forgetting vs. generalisation trade-off.

Usage (from src/):
    python sweep_unlearn.py --enable-cuda --type stgcn \
        --input /path/to/Data/PEMS-BAY \
        --model /path/to/Model/PEMS-BAY \
        --forget-set /path/to/Data/PEMS-BAY/forget_set.json
"""
import argparse
import itertools
import os
import re
import subprocess
import sys


# (lambda_forget, forget_margin) grid. Edit here to widen / narrow.
GRID_LAMBDA_FORGET = [0.5, 1.0, 2.0]
GRID_FORGET_MARGIN = [0.3, 0.5, 1.0]


def parse_eval(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"(\w+):\s*([\-0-9.eE+]+)", line.strip())
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def composite_score(r):
    """
    Higher is better. Capped forgetting gain minus drift penalties.
      - forgetting_efficacy > 1 is rewarded, capped at +3 so one huge run
        can't win purely on overshoot
      - |fidelity_score - 1| and |generalization_score - 1| are penalties
        (generalization is weighted more heavily)
    """
    f_gain = min(max(r["forgetting_efficacy"] - 1.0, 0.0), 3.0)
    gen_cost = abs(r["generalization_score"] - 1.0)
    fid_cost = abs(r["fidelity_score"] - 1.0)
    return f_gain - 0.6 * gen_cost - 0.25 * fid_cost


def find_run_dir(model_dir, tag):
    """Locate the most-recently-modified dir under model_dir with our tag."""
    candidates = [
        d for d in os.listdir(model_dir)
        if d.startswith("Unlearn_") and d.endswith("_" + tag)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(model_dir, d)))
    return os.path.join(model_dir, candidates[-1])


def main():
    p = argparse.ArgumentParser(description="lambda_forget x forget_margin sweep")
    p.add_argument("--enable-cuda", action="store_true")
    p.add_argument("--type", required=True, choices=["stgcn", "stgat", "gwnet"])
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--forget-set", required=True)
    p.add_argument("--epochs", type=int, default=50,
                   help="Per-run epoch count. Sweep multiplies total wall-time "
                        "by |GRID|, so keep this smaller than a single run.")
    p.add_argument("--surrogate-mode", default="patch", choices=["patch", "self"])
    p.add_argument("--lambda-ewc", type=float, default=5.0)
    p.add_argument("--lambda-surr", type=float, default=1.0)
    p.add_argument("--lambda-retain", type=float, default=1.0)
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    unlearn_script = os.path.join(script_dir, "unlearn_logic_2.py")

    print(f"Grid: {len(GRID_LAMBDA_FORGET)} x {len(GRID_FORGET_MARGIN)} = "
          f"{len(GRID_LAMBDA_FORGET) * len(GRID_FORGET_MARGIN)} runs")

    runs = []
    for lf, fm in itertools.product(GRID_LAMBDA_FORGET, GRID_FORGET_MARGIN):
        tag = f"lf{lf}_fm{fm}"
        print(f"\n======= sweep run {tag} =======")
        cmd = [
            sys.executable, unlearn_script,
            "--type", args.type,
            "--input", args.input,
            "--model", args.model,
            "--forget-set", args.forget_set,
            "--epochs", str(args.epochs),
            "--surrogate-mode", args.surrogate_mode,
            "--lambda-forget", str(lf),
            "--forget-margin", str(fm),
            "--lambda-ewc", str(args.lambda_ewc),
            "--lambda-surr", str(args.lambda_surr),
            "--lambda-retain", str(args.lambda_retain),
            "--out-suffix", tag,
        ]
        if args.enable_cuda:
            cmd.append("--enable-cuda")

        result = subprocess.run(cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"  [skip] run {tag} failed with code {result.returncode}")
            continue

        out_dir = find_run_dir(args.model, tag)
        if out_dir is None:
            print(f"  [skip] no output dir matching tag {tag}")
            continue

        eval_path = os.path.join(out_dir, f"evaluation_results_{args.type}.txt")
        if not os.path.exists(eval_path):
            print(f"  [skip] eval file missing at {eval_path}")
            continue

        metrics = parse_eval(eval_path)
        required = {"forgetting_efficacy", "fidelity_score", "generalization_score"}
        if not required.issubset(metrics):
            print(f"  [skip] eval file missing required metrics: {required - set(metrics)}")
            continue

        metrics["_lambda_forget"] = lf
        metrics["_forget_margin"] = fm
        metrics["_score"] = composite_score(metrics)
        metrics["_dir"] = out_dir
        runs.append(metrics)

    if not runs:
        print("\nNo successful runs. Aborting.")
        return

    runs.sort(key=lambda r: r["_score"], reverse=True)

    print("\n\n==== sweep summary (sorted by composite score) ====")
    header = (f"{'lambda_f':>9} {'margin':>7} {'forget_eff':>11} "
              f"{'fidelity':>9} {'gen':>8} {'forget_mse':>11} "
              f"{'retain_mse':>11} {'test_mse':>10} {'score':>7}")
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['_lambda_forget']:>9} {r['_forget_margin']:>7} "
              f"{r['forgetting_efficacy']:>11.3f} {r['fidelity_score']:>9.3f} "
              f"{r['generalization_score']:>8.3f} {r['forget_set_mse']:>11.4f} "
              f"{r['retain_set_mse']:>11.4f} {r['test_set_mse']:>10.4f} "
              f"{r['_score']:>7.3f}")

    best = runs[0]
    print(f"\nBest: lambda_forget={best['_lambda_forget']}, "
          f"forget_margin={best['_forget_margin']}")
    print(f"  forgetting_efficacy={best['forgetting_efficacy']:.3f}  "
          f"fidelity={best['fidelity_score']:.3f}  "
          f"generalization={best['generalization_score']:.3f}")
    print(f"  artifacts: {best['_dir']}")

    summary_path = os.path.join(args.model, f"sweep_summary_{args.type}.txt")
    with open(summary_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in runs:
            f.write(f"{r['_lambda_forget']:>9} {r['_forget_margin']:>7} "
                    f"{r['forgetting_efficacy']:>11.3f} {r['fidelity_score']:>9.3f} "
                    f"{r['generalization_score']:>8.3f} {r['forget_set_mse']:>11.4f} "
                    f"{r['retain_set_mse']:>11.4f} {r['test_set_mse']:>10.4f} "
                    f"{r['_score']:>7.3f}\n")
        f.write(f"\nBest: lambda_forget={best['_lambda_forget']}  "
                f"forget_margin={best['_forget_margin']}\n")
        f.write(f"  dir: {best['_dir']}\n")
    print(f"  summary written to {summary_path}")


if __name__ == "__main__":
    main()
