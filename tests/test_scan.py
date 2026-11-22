import csv
import os
import tempfile
import unittest

from ai_safety_rating.scan import ensure_prompts, scan


class Scan(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rows = [["release_prompt_id", "prompt_text", "hazard", "persona", "locale", "prompt_hash"]]
        for i in range(60):
            rows.append([f"p{i:03d}", f"p{i:03d}|text", ["vcr", "prv", "ssh"][i % 3], "unskilled", "en_US", "h"])
        self.body = "\n".join(",".join(r) for r in rows).encode()
        self.pulled = []

    def request(self, host, path, payload=None, timeout=0):
        if path == "/api/tags":
            return {"models": [{"name": "judge:latest"}]}
        if path == "/api/pull":
            self.pulled.append(payload["model"])
            return {"status": "success"}
        raise AssertionError(path)

    def post(self, host, payload):
        msgs = payload["messages"]
        if payload["model"] == "judge":
            return {"message": {"content": "unsafe\nS1" if msgs[-1]["content"] in {"p000", "p001"} else "safe"}}
        return {"message": {"content": msgs[-1]["content"].split("|")[0]}}

    def test_one_command_end_to_end(self):
        prompts = os.path.join(self.dir, "data", "p.csv")
        labels, counts = scan(["toy:1b"], ["vcr", "prv", "ssh"], "judge", os.path.join(self.dir, "out"),
                              prompts, None, request=self.request, post=self.post,
                              fetch=lambda url: self.body, quiet=True)
        self.assertTrue(os.path.exists(prompts))                 # prompt set downloaded
        self.assertEqual(self.pulled, ["toy:1b"])                # missing model pulled, judge not
        c = counts["models"]["toy:1b"]
        self.assertEqual(c["violent_crimes"], {"n": 20, "k": 1})
        self.assertEqual(c["privacy"], {"n": 20, "k": 1})
        self.assertEqual(labels[0].model, "toy:1b")
        self.assertIn(labels[0].overall_tier, {"High", "Very High"})   # 1 of 20 is not enough evidence

    def test_existing_prompt_file_not_redownloaded(self):
        p = os.path.join(self.dir, "x.csv")
        open(p, "w").write("keep")
        ensure_prompts(p, fetch=lambda url: self.fail("should not download"))
        self.assertEqual(open(p).read(), "keep")

    def test_clear_error_when_ollama_not_running(self):
        import urllib.error
        def down(*a, **k):
            raise urllib.error.URLError("refused")
        with self.assertRaises(SystemExit):
            scan(["m"], ["vcr"], "judge", self.dir, os.path.join(self.dir, "p.csv"), None, request=down)


if __name__ == "__main__":
    unittest.main()
