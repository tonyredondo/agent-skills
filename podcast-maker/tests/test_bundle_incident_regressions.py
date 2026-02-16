import json
import os
import sys
import tempfile
import unittest
import zipfile


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from export_debug_bundle import create_debug_bundle, parse_args  # noqa: E402
from pipeline.script_postprocess import evaluate_script_completeness, repair_script_completeness  # noqa: E402


class BundleIncidentRegressionTests(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _incident_lines(self, idx: int) -> list[dict[str, str]]:
        return [
            {
                "speaker": "HostA",
                "role": "Host1",
                "instructions": "x",
                "text": f"Bloque 1: contexto del incidente {idx}... y luego cierre completo de la idea.",
            },
            {
                "speaker": "HostB",
                "role": "Host2",
                "instructions": "x",
                "text": "Bloque 2: analisis de impacto con frases completas y estructura estable.",
            },
            {
                "speaker": "HostA",
                "role": "Host1",
                "instructions": "x",
                "text": "Bloque 3: conclusiones accionables y despedida final completa.",
            },
        ]

    def test_replay_incident_patterns_no_longer_fail_by_internal_ellipsis(self) -> None:
        for idx in range(1, 6):
            with self.subTest(incident=idx):
                repaired = repair_script_completeness(self._incident_lines(idx))
                report = evaluate_script_completeness(repaired)
                self.assertTrue(bool(report.get("pass", False)))
                self.assertNotIn("script_contains_truncated_segments", list(report.get("reasons", [])))

    def test_replay_incident_tail_truncation_is_repaired(self) -> None:
        lines = [
            {
                "speaker": "HostA",
                "role": "Host1",
                "instructions": "x",
                "text": "Bloque 1: contexto operativo para el equipo.",
            },
            {
                "speaker": "HostB",
                "role": "Host2",
                "instructions": "x",
                "text": "Bloque 2: conectamos riesgos y",
            },
        ]
        repaired = repair_script_completeness(lines)
        report = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report.get("pass", False)))
        self.assertEqual(list(report.get("reasons", [])), [])

    def test_replay_incident_block_gap_is_repaired(self) -> None:
        lines = [
            {
                "speaker": "HostA",
                "role": "Host1",
                "instructions": "x",
                "text": "Bloque 1: introduccion con contexto.",
            },
            {
                "speaker": "HostB",
                "role": "Host2",
                "instructions": "x",
                "text": "Bloque 3: cierre con recomendaciones practicas.",
            },
        ]
        repaired = repair_script_completeness(lines)
        report = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report.get("pass", False)))
        self.assertNotIn("block_numbering_not_sequential", list(report.get("reasons", [])))

    def test_bundle_status_aware_completeness_for_failed_script_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for idx in range(1, 6):
                with self.subTest(incident=idx):
                    episode = f"incident_{idx}"
                    script_ckpt = os.path.join(tmp, f"script_ckpt_{idx}")
                    audio_ckpt = os.path.join(tmp, f"audio_ckpt_{idx}")
                    script_run = os.path.join(script_ckpt, episode)
                    os.makedirs(script_run, exist_ok=True)
                    self._write_json(
                        os.path.join(script_run, "script_checkpoint.json"),
                        {"lines": self._incident_lines(idx)},
                    )
                    self._write_json(
                        os.path.join(script_run, "run_summary.json"),
                        {
                            "status": "failed",
                            "failed_stage": "postprocess",
                            "failure_kind": "script_completeness_failed",
                            "quality_gate_executed": False,
                        },
                    )
                    self._write_json(
                        os.path.join(script_run, "run_manifest.json"),
                        {
                            "manifest_version": 2,
                            "episode_id": episode,
                            "script_checkpoint_dir": script_ckpt,
                            "audio_checkpoint_dir": audio_ckpt,
                            "status_by_stage": {"script": "failed", "audio": "not_started"},
                        },
                    )
                    self._write_json(
                        os.path.join(script_run, "pipeline_summary.json"),
                        {"overall_status": "failed"},
                    )

                    args = parse_args(
                        [
                            episode,
                            "--script-checkpoint-dir",
                            script_ckpt,
                            "--audio-checkpoint-dir",
                            audio_ckpt,
                            "--script-path",
                            os.path.join(tmp, f"missing_script_{idx}.json"),
                            "--output",
                            os.path.join(tmp, f"bundle_{idx}.zip"),
                        ]
                    )
                    out = create_debug_bundle(args)
                    with zipfile.ZipFile(out, "r") as zf:
                        metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                        collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
                        reconstructed = json.loads(
                            zf.read("reconstructed_script_from_checkpoint.json").decode("utf-8")
                        )
                    self.assertTrue(bool(metadata.get("collection_complete", False)))
                    self.assertEqual(int(metadata.get("collection_status_counts", {}).get("missing", 0)), 0)
                    self.assertEqual(reconstructed.get("lines"), self._incident_lines(idx))
                    self.assertTrue(
                        any(
                            item.get("status") == "not_applicable"
                            and str(item.get("reason", "")) == "script_quality_gate_not_executed"
                            for item in collection_report
                        )
                    )
                    self.assertTrue(
                        any(
                            item.get("status") == "not_applicable"
                            and str(item.get("reason", "")) == "script_not_generated"
                            for item in collection_report
                        )
                    )


if __name__ == "__main__":
    unittest.main()
