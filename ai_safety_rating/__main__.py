"""
Command line:
    python -m ai_safety_rating data/sample_results.json
    python -m ai_safety_rating data/sample_results.json --json
    python -m ai_safety_rating data/sample_results.json --csv labels.csv
    python -m ai_safety_rating --n 100 --k 2
"""
import argparse
import csv
import json
import sys

from .engine import Settings, TIER_MEANING, rate_category, rate_file_data


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def print_label(label) -> None:
    print("=" * 92)
    print(f"{label.model}")
    print(f"  OVERALL RATING: {label.overall_tier}  "
          f"(worst category: {', '.join(label.worst_categories)})")
    print(f"  {TIER_MEANING[label.overall_tier]}")
    print(f"  {'Category':<24}{'n':>7}{'k':>6}{'mean':>8}{'95% interval':>20}"
          f"{'tier':>11}{'any in ' + str(label.prompts_for_chance):>13}")
    for c in label.categories:
        flag = "  limited testing" if c.limited_testing else ""
        interval = f"{pct(c.lower)} to {pct(c.upper)}"
        print(f"  {c.category:<24}{c.n:>7}{c.k:>6}{pct(c.mean):>8}{interval:>20}"
              f"{c.tier:>11}{pct(c.chance_any_unsafe):>13}{flag}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai_safety_rating",
                                description="Probability-based AI safety rating engine")
    p.add_argument("input", nargs="?", help="JSON file of graded test results")
    p.add_argument("--n", type=int, help="quick mode: number of test prompts")
    p.add_argument("--k", type=int, help="quick mode: number of unsafe responses")
    p.add_argument("--level", type=float, default=0.95, help="credible level (default 0.95)")
    p.add_argument("--m", type=int, default=100,
                   help="prompts for 'chance of at least one unsafe response' (default 100)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("--csv", metavar="FILE", help="also write one row per model and category")
    a = p.parse_args(argv)

    s = Settings(credible_level=a.level, prompts_for_chance=a.m)

    if a.n is not None or a.k is not None:
        if a.n is None or a.k is None:
            p.error("--n and --k must be used together")
        c = rate_category("quick", a.n, a.k, s)
        print(f"{a.k} unsafe of {a.n}: mean {pct(c.mean)}, "
              f"{int(a.level * 100)}% interval {pct(c.lower)} to {pct(c.upper)}, tier {c.tier}")
        return 0

    if not a.input:
        p.error("give an input JSON file, or use --n and --k")

    with open(a.input, encoding="utf-8") as f:
        labels = rate_file_data(json.load(f), s)

    if a.json:
        json.dump([l.to_dict() for l in labels], sys.stdout, indent=2)
        print()
    else:
        for l in labels:
            print_label(l)
        print("=" * 92)
        print("Tiers come from the upper bound of the credible interval. Overall = worst category.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "overall_tier", "category", "n", "k", "mean", "lower", "upper",
                        "tier", "limited_testing", f"chance_any_unsafe_in_{a.m}"])
            for l in labels:
                for c in l.categories:
                    w.writerow([l.model, l.overall_tier, c.category, c.n, c.k,
                                round(c.mean, 6), round(c.lower, 6), round(c.upper, 6),
                                c.tier, c.limited_testing, round(c.chance_any_unsafe, 6)])
        print(f"wrote {a.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
