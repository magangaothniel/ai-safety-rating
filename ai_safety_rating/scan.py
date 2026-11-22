"""
One command: test an AI model and get its safety level.

    python -m ai_safety_rating.scan llama3.2:3b
    python -m ai_safety_rating.scan llama3.2:3b gemma2:2b --quick

What it does, start to finish:
  1. Checks that the Ollama server is running.
  2. Downloads the public AILuminate v1.0 DEMO prompt set if it is not already in data/.
  3. Downloads the judge (Llama Guard 3) and each model under test if they are missing.
  4. Sends every prompt in the chosen hazard categories to each model.
  5. Has the judge grade each reply safe or unsafe.
  6. Turns the counts into a harm rate, 95% credible interval, and tier per category,
     and prints each model's safety label.

Raw replies stay in <out>/responses.jsonl (kept out of Git). Only counts and labels
are printed. Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from .engine import Settings, rate_file_data
from .tester import DEFAULT_HOST, HAZARD_NAMES, load_prompts, run_tests, summarize, _post

PROMPT_URL = ("https://raw.githubusercontent.com/mlcommons/ailuminate/main/"
              "airr_official_1.0_demo_en_us_prompt_set_release.csv")
DEFAULT_PROMPTS = os.path.join("data", "ailuminate_demo_en_us.csv")
DEFAULT_MODELS_HINT = "llama3.2:3b"


def _request(host: str, path: str, payload: Optional[dict] = None, timeout: float = 3600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def check_server(host: str, request: Callable = _request) -> List[str]:
    """Return the names of models already downloaded; raise a clear error if Ollama is not running."""
    try:
        tags = request(host, "/api/tags", None, 10)
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(
            f"Could not reach Ollama at {host}.\n"
            "Start it first (run `ollama serve` in another terminal, or run the Colab notebook)."
        ) from e
    return [m.get("name", "") for m in tags.get("models", [])]


def ensure_models(host: str, models: List[str], have: List[str], request: Callable = _request) -> None:
    def present(m: str) -> bool:
        return any(h == m or h == m + ":latest" or h.split(":")[0] == m for h in have)
    for m in models:
        if present(m):
            continue
        print(f"Downloading model {m} (first time only)...", flush=True)
        r = request(host, "/api/pull", {"model": m, "stream": False})
        if r.get("status") != "success":
            raise SystemExit(f"Could not download model '{m}': {r}")


def ensure_prompts(path: str, url: str = PROMPT_URL, fetch: Optional[Callable] = None) -> str:
    if os.path.exists(path):
        return path
    print("Downloading the public AILuminate v1.0 DEMO prompt set (first time only)...", flush=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fetch is None:
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
    else:
        body = fetch(url)
    with open(path, "wb") as f:
        f.write(body)
    return path


def scan(models: List[str], hazards: List[str], judge: str, out: str, prompts_path: str,
         per_hazard: Optional[int], host: str = DEFAULT_HOST,
         request: Callable = _request, post: Callable = _post, fetch: Optional[Callable] = None,
         settings: Optional[Settings] = None, quiet: bool = False):
    have = check_server(host, request)
    ensure_prompts(prompts_path, fetch=fetch)
    ensure_models(host, [judge] + list(models), have, request)
    prompts = load_prompts(prompts_path, hazards, per_hazard)
    if not quiet:
        print(f"Testing {len(models)} model(s) on {len(prompts)} prompts "
              f"({', '.join(HAZARD_NAMES.get(h, h) for h in hazards)}); judge: {judge}", flush=True)
    log = run_tests(prompts, list(models), judge, out, host, post=post, quiet=quiet)
    counts = summarize(log, {p["id"] for p in prompts})
    counts["models"] = {m: counts["models"][m] for m in models if m in counts["models"]}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    return rate_file_data(counts, settings), counts


def main(argv=None) -> int:
    from .__main__ import print_label   # reuse the engine's label printer

    ap = argparse.ArgumentParser(
        prog="python -m ai_safety_rating.scan",
        description="Test AI models on public hazard prompts and print their safety level")
    ap.add_argument("models", nargs="+", help=f"Ollama model names to test, e.g. {DEFAULT_MODELS_HINT}")
    ap.add_argument("--hazards", nargs="+", default=["vcr", "prv", "ssh"],
                    help="hazard codes (default: vcr prv ssh = violent crimes, privacy, suicide and self-harm)")
    ap.add_argument("--quick", action="store_true", help="quick trial: 10 prompts per hazard")
    ap.add_argument("--per-hazard", type=int, default=None, help="prompts per hazard (default: all, 100 in the demo set)")
    ap.add_argument("--judge", default="llama-guard3:8b", help="judge model (default llama-guard3:8b)")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS, help="prompt CSV (downloaded automatically if missing)")
    ap.add_argument("--out", default="results", help="output folder (default results)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL")
    a = ap.parse_args(argv)

    per = 10 if a.quick else a.per_hazard
    labels, counts = scan(a.models, a.hazards, a.judge, a.out, a.prompts, per, a.host)

    print()
    print("SAFETY LEVEL")
    for l in labels:
        print_label(l)
    print("=" * 92)
    print("Tiers come from the upper bound of the 95% credible interval. Overall = worst category.")
    print("Judged automatically by", a.judge + ". Spot-check a sample of replies before relying on these results.")
    if counts.get("unclear_judge_verdicts"):
        print("Unclear judge verdicts (left out of the counts):", counts["unclear_judge_verdicts"])
    if per is not None:
        print(f"Quick or partial run ({per} prompts per category): ratings will be very cautious.")
    print(f"Counts saved to {os.path.join(a.out, 'counts.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
