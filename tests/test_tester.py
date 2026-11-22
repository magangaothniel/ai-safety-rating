import csv
import json
import os
import tempfile
import unittest

from ai_safety_rating import rate_file_data
from ai_safety_rating.tester import load_prompts, parse_verdict, run_tests, summarize


def fake_server(unsafe_ids):
    """Stands in for Ollama: the test model echoes the prompt id, the judge flags chosen ids."""
    def post(host, payload):
        msgs = payload["messages"]
        if payload["model"].startswith("judge"):
            reply = msgs[-1]["content"]
            return {"message": {"content": "unsafe\nS1" if reply in unsafe_ids else "safe"}}
        return {"message": {"content": msgs[-1]["content"].split("|")[0]}}
    return post


class Tester(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.csv = os.path.join(self.dir, "p.csv")
        with open(self.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Release_prompt_id", "Prompt_text", "Hazard", "Persona", "Locale", "Prompt_hash"])
            for i in range(30):
                hz = ["vcr", "prv", "ssh"][i % 3]
                w.writerow([f"id{i:03d}", f"id{i:03d}|text", hz, "normal", "en_US", "h"])

    def test_parse_verdict(self):
        self.assertEqual(parse_verdict("safe"), (False, ""))
        self.assertEqual(parse_verdict("unsafe\nS2"), (True, "S2"))
        self.assertEqual(parse_verdict("  Unsafe \n S10 "), (True, "S10"))
        self.assertEqual(parse_verdict("I cannot classify"), (None, ""))

    def test_load_and_sample(self):
        p = load_prompts(self.csv, ["vcr", "prv"])
        self.assertEqual(len(p), 20)
        s1 = load_prompts(self.csv, ["vcr"], per_hazard=4)
        s2 = load_prompts(self.csv, ["vcr"], per_hazard=4)
        self.assertEqual([r["id"] for r in s1], [r["id"] for r in s2])   # fixed sample
        with self.assertRaises(ValueError):
            load_prompts(self.csv, ["hte"])

    def test_end_to_end_counts_and_rating(self):
        prompts = load_prompts(self.csv, ["vcr", "prv", "ssh"])
        post = fake_server({"id000", "id003", "id001"})      # 2 vcr, 1 prv unsafe
        out = os.path.join(self.dir, "out")
        log = run_tests(prompts, ["m1"], "judge", out, post=post, quiet=True)
        counts = summarize(log)
        c = counts["models"]["m1"]
        self.assertEqual(c["violent_crimes"], {"n": 10, "k": 2})
        self.assertEqual(c["privacy"], {"n": 10, "k": 1})
        self.assertEqual(c["suicide_and_self_harm"], {"n": 10, "k": 0})
        label = rate_file_data(counts)[0]
        self.assertTrue(all(x.limited_testing for x in label.categories))

    def test_resume_skips_finished(self):
        prompts = load_prompts(self.csv, ["vcr"])
        out = os.path.join(self.dir, "out2")
        calls = []
        base = fake_server(set())
        def counting(host, payload):
            calls.append(1)
            return base(host, payload)
        run_tests(prompts, ["m1"], "judge", out, post=counting, quiet=True)
        first = len(calls)
        run_tests(prompts, ["m1"], "judge", out, post=counting, quiet=True)
        self.assertEqual(len(calls), first)                  # nothing re-run
        with open(os.path.join(out, "responses.jsonl")) as f:
            self.assertEqual(sum(1 for _ in f), 10)


if __name__ == "__main__":
    unittest.main()
