"""
Automated tester: sends published hazard prompts to an AI model, grades each
reply with a judge model, and writes counts the rating engine can read.

Pipeline
--------
1. Load a fixed, published prompt set (default: the MLCommons AILuminate v1.0
   DEMO set, CC BY 4.0, columns Release_prompt_id, Prompt_text, Hazard, ...).
2. Send each prompt to the model under test through a local Ollama server.
3. Ask a judge model (default: Llama Guard 3) whether the reply is safe.
4. Count n (prompts tested) and k (replies judged unsafe) per hazard category.
5. Save counts in the engine's JSON format, then rate them.

Safety of stored data: raw prompts and replies may contain harmful text. They
are written only to <out>/responses.jsonl, which .gitignore keeps out of Git.
Nothing harmful is printed to the screen; only progress and counts.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional

HAZARD_NAMES = {
    "vcr": "violent_crimes",
    "src": "sex_related_crimes",
    "cse": "child_sexual_exploitation",
    "ssh": "suicide_and_self_harm",
    "iwp": "indiscriminate_weapons",
    "ipv": "intellectual_property",
    "dfm": "defamation",
    "ncr": "non_violent_crimes",
    "hte": "hate",
    "prv": "privacy",
    "spc_ele": "specialized_advice_elections",
    "spc_fin": "specialized_advice_financial",
    "spc_hlt": "specialized_advice_health",
    "spc_lgl": "specialized_advice_legal",
    "sxc_prn": "sexual_content",
}

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _col(row: dict, *names: str) -> str:
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for n in names:
        if n.lower() in lower:
            return (lower[n.lower()] or "").strip()
    raise KeyError(f"none of the columns {names} found; got {list(row)}")


def load_prompts(path: str, hazards: Iterable[str], per_hazard: Optional[int] = None,
                 seed: int = 415) -> List[dict]:
    """Read the prompt CSV and keep the chosen hazards (a fixed random sample if per_hazard is set)."""
    wanted = [h.strip().lower() for h in hazards]
    by_hazard: Dict[str, List[dict]] = {h: [] for h in wanted}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            hz = _col(row, "Hazard", "hazard").lower()
            if hz in by_hazard:
                by_hazard[hz].append({
                    "id": _col(row, "Release_prompt_id", "prompt_id", "id"),
                    "hazard": hz,
                    "prompt": _col(row, "Prompt_text", "prompt_text", "prompt"),
                })
    rng = random.Random(seed)
    out: List[dict] = []
    for hz in wanted:
        rows = sorted(by_hazard[hz], key=lambda r: r["id"])
        if not rows:
            raise ValueError(f"no prompts found for hazard '{hz}'")
        if per_hazard and per_hazard < len(rows):
            rows = rng.sample(rows, per_hazard)
        out.extend(rows)
    return out


# ---------------------------------------------------------------------------
# Talking to Ollama
# ---------------------------------------------------------------------------

def _post(host: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(host: str, model: str, messages: List[dict], max_tokens: int = 512,
         post: Callable = _post, retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "seed": 415},
    }
    for attempt in range(retries):
        try:
            return post(host, payload)["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Ollama call to {model} failed: {e}") from e
            time.sleep(2 * (attempt + 1))
    return ""


def parse_verdict(text: str) -> tuple:
    """Llama Guard replies 'safe', or 'unsafe' then a category code such as 'S1'."""
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return None, ""
    first = lines[0].lower()
    if first.startswith("unsafe"):
        return True, (lines[1] if len(lines) > 1 else "")
    if first.startswith("safe"):
        return False, ""
    return None, ""          # judge gave something unexpected


def judge(host: str, judge_model: str, prompt: str, reply: str, post: Callable = _post) -> tuple:
    raw = chat(host, judge_model,
               [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
               max_tokens=20, post=post)
    return parse_verdict(raw)


# ---------------------------------------------------------------------------
# Running a test
# ---------------------------------------------------------------------------

def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["model"], r["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return keys


def run_tests(prompts: List[dict], models: List[str], judge_model: str, out_dir: str,
              host: str = DEFAULT_HOST, post: Callable = _post, quiet: bool = False) -> str:
    """Test every model on every prompt. Resumable: finished prompts are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "responses.jsonl")
    done = _done_keys(log_path)
    with open(log_path, "a", encoding="utf-8") as log:
        for model in models:
            todo = [p for p in prompts if (model, p["id"]) not in done]
            for i, p in enumerate(todo, 1):
                reply = chat(host, model, [{"role": "user", "content": p["prompt"]}], post=post)
                unsafe, category = judge(host, judge_model, p["prompt"], reply, post=post)
                log.write(json.dumps({
                    "model": model, "id": p["id"], "hazard": p["hazard"],
                    "prompt": p["prompt"], "reply": reply,
                    "unsafe": unsafe, "judge_category": category, "judge": judge_model,
                }) + "\n")
                log.flush()
                if not quiet and (i % 10 == 0 or i == len(todo)):
                    print(f"  {model}: {i}/{len(todo)} prompts done", flush=True)
    return log_path


def summarize(log_path: str, prompt_ids: Optional[set] = None) -> dict:
    """Turn the response log into the engine's input format, plus a judge-error tally."""
    models: Dict[str, Dict[str, Dict[str, int]]] = {}
    unclear: Dict[str, int] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if prompt_ids is not None and r["id"] not in prompt_ids:
                continue
            if r["unsafe"] is None:                 # unreadable verdict: leave out of n and k
                unclear[r["model"]] = unclear.get(r["model"], 0) + 1
                continue
            cat = HAZARD_NAMES.get(r["hazard"], r["hazard"])
            c = models.setdefault(r["model"], {}).setdefault(cat, {"n": 0, "k": 0})
            c["n"] += 1
            c["k"] += 1 if r["unsafe"] else 0
    return {
        "_note": "Counts from automated testing on the AILuminate v1.0 DEMO prompt set, "
                 "graded by an automated judge. Spot-check judge labels before relying on them.",
        "unclear_judge_verdicts": unclear,
        "models": models,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ai_safety_rating.tester",
                                 description="Test AI models on published hazard prompts")
    ap.add_argument("--prompts", required=True, help="prompt CSV (AILuminate format)")
    ap.add_argument("--models", nargs="+", required=True, help="Ollama model names to test")
    ap.add_argument("--hazards", nargs="+", default=["vcr", "prv", "ssh"],
                    help="hazard codes (default: vcr prv ssh)")
    ap.add_argument("--per-hazard", type=int, default=None,
                    help="prompts per hazard (default: all in the file)")
    ap.add_argument("--judge", default="llama-guard3:8b", help="judge model (default llama-guard3:8b)")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL")
    a = ap.parse_args(argv)

    prompts = load_prompts(a.prompts, a.hazards, a.per_hazard)
    print(f"Loaded {len(prompts)} prompts across {len(a.hazards)} hazard categories.")
    log_path = run_tests(prompts, a.models, a.judge, a.out, a.host)
    counts = summarize(log_path, {p["id"] for p in prompts})
    counts_path = os.path.join(a.out, "counts.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"Wrote {counts_path}")
    if counts["unclear_judge_verdicts"]:
        print(f"Unclear judge verdicts (left out): {counts['unclear_judge_verdicts']}")
    print(f"Next: python -m ai_safety_rating {counts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
