import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_script  # noqa: E402
import run_golden_pipeline as runner  # noqa: E402
from pipeline.golden_gate import evaluate_golden_suite  # noqa: E402
from pipeline.golden_metrics import compute_script_metrics  # noqa: E402


class _FakeScriptClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0
        self.script_retries_total = 0
        self.script_json_parse_failures = 0
        self._seq = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self._seq += 1
        return {
            "lines": [
                {
                    "speaker": "Ana",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": f"Bloque {self._seq} con ideas utiles y contexto practico para la audiencia.",
                },
                {
                    "speaker": "Luis",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": f"Seguimos con ejemplos concretos en el bloque {self._seq} para mantener claridad y utilidad.",
                },
            ]
        }

    def generate_freeform_text(self, *, prompt, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return "Resumen breve con puntos clave."


class GoldenPipelineE2ETests(unittest.TestCase):
    def test_runner_plus_gate_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write(("contenido de prueba " * 200).strip())

            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "case_name": "e2e_case",
                            "profile": "short",
                            "source_path": source,
                            "min_words": 20,
                            "max_words": 60,
                        }
                    ],
                    f,
                )

            candidate_dir = os.path.join(tmp, "candidates")
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=candidate_dir,
                debug=False,
                stop_on_error=False,
                report_json="",
            )

            def fake_subprocess_run(cmd, capture_output, text, check):  # noqa: ANN001
                argv = cmd[2:]
                with mock.patch.object(make_script.OpenAIClient, "from_configs", return_value=_FakeScriptClient()):
                    rc = make_script.main(argv)
                return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="")

            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run", side_effect=fake_subprocess_run):
                    rc = runner.main()
            self.assertEqual(rc, 0)

            candidate_path = os.path.join(candidate_dir, "e2e_case.json")
            self.assertTrue(os.path.exists(candidate_path))
            with open(candidate_path, "r", encoding="utf-8") as f:
                candidate_payload = json.load(f)

            baseline_path = os.path.join(tmp, "baseline_metrics.json")
            baseline = {"e2e_case": compute_script_metrics(candidate_payload).to_dict()}
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            gate = evaluate_golden_suite(
                baseline_path=baseline_path,
                candidate_dir=candidate_dir,
            )
            self.assertTrue(gate["overall_pass"], msg=str(gate))


if __name__ == "__main__":
    unittest.main()

