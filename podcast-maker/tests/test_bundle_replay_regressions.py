import json
import os
import sys
import tempfile
import unittest
import zipfile
from typing import Dict, Tuple


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from export_debug_bundle import create_debug_bundle, parse_args  # noqa: E402


class BundleReplayRegressionTests(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _build_case(
        self,
        *,
        root: str,
        bundle_id: str,
        scenario: str,
        force_episode_mismatch: bool = False,
    ) -> Tuple[str, str, str, str]:
        episode_id = str(bundle_id)
        script_ckpt = os.path.join(root, "script_ckpt")
        audio_ckpt = os.path.join(root, "audio_ckpt")
        run_episode = f"{episode_id}_other" if force_episode_mismatch else episode_id
        script_run = os.path.join(script_ckpt, run_episode)
        manifest_run = os.path.join(script_ckpt, episode_id)
        script_path = os.path.join(root, f"{episode_id}_script.json")
        source_path = os.path.join(root, f"{episode_id}_source.txt")

        quality_gate_executed = scenario in {"editorial_warn_only", "quality_interrupted"}
        quality_stage_started = scenario in {"editorial_warn_only", "quality_interrupted"}
        quality_stage_finished = scenario == "editorial_warn_only"
        quality_stage_interrupted = scenario == "quality_interrupted"
        if scenario == "editorial_warn_only":
            script_status = "completed"
            failure_kind = None
        elif scenario == "quality_interrupted":
            script_status = "interrupted"
            failure_kind = "interrupted"
        elif scenario == "openai_empty_output_failed":
            script_status = "failed"
            failure_kind = "openai_empty_output"
        elif scenario == "root_mismatch":
            script_status = "failed"
            failure_kind = "invalid_schema"
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

        script_block: Dict[str, object] = {
            "run_summary_path": os.path.join(script_run, "run_summary.json"),
        }
        if scenario in {"editorial_warn_only", "quality_interrupted"}:
            script_block["quality_report_path"] = os.path.join(script_run, "quality_report.json")
        if scenario == "quality_interrupted":
            script_block["quality_report_initial_path"] = os.path.join(
                script_run,
                "quality_report_initial.json",
            )
            script_block["quality_stage_started"] = True
            script_block["quality_stage_finished"] = False
            script_block["quality_stage_interrupted"] = True

        manifest_payload = {
            "manifest_version": 2,
            "episode_id": run_episode,
            "script_output_path": script_path,
            "script_checkpoint_dir": script_ckpt,
            "audio_checkpoint_dir": audio_ckpt,
            "status_by_stage": {"script": script_status, "audio": "not_started"},
            "script": script_block,
            "audio": {},
        }
        self._write_json(os.path.join(manifest_run, "run_manifest.json"), manifest_payload)
        if force_episode_mismatch:
            self._write_json(os.path.join(script_run, "run_manifest.json"), manifest_payload)
        self._write_json(
            os.path.join(script_run, "script_checkpoint.json"),
            {
                "status": script_status,
                "lines": [
                    {
                        "speaker": "Ana",
                        "role": "Host1",
                        "instructions": "x",
                        "text": "Bloque 1 con contenido estable y trazable.",
                    }
                ],
            },
        )
        self._write_json(
            os.path.join(script_run, "run_summary.json"),
            {
                "episode_id": run_episode,
                "status": script_status,
                "failure_kind": failure_kind,
                "quality_gate_executed": quality_gate_executed,
                "quality_stage_started": quality_stage_started,
                "quality_stage_finished": quality_stage_finished,
                "quality_stage_interrupted": quality_stage_interrupted,
                "script_gate_action_effective": "enforce" if quality_gate_executed else "off",
            },
        )
        self._write_json(
            os.path.join(script_run, "pipeline_summary.json"),
            {
                "overall_status": script_status,
                "episode_id": run_episode,
                "status_by_stage": {"script": script_status, "audio": "not_started"},
            },
        )

        if scenario == "editorial_warn_only":
            self._write_json(
                os.path.join(script_run, "quality_report_initial.json"),
                {
                    "status": "failed",
                    "pass": False,
                    "quality_report_phase": "initial",
                    "quality_stage_started": True,
                    "quality_stage_finished": False,
                    "quality_stage_interrupted": False,
                },
            )
            self._write_json(
                os.path.join(script_run, "quality_report.json"),
                {
                    "status": "passed",
                    "pass": True,
                    "hard_fail_eligible": False,
                    "editorial_warn_only": True,
                    "reasons": ["overall_score_below_threshold"],
                },
            )
        elif scenario == "quality_interrupted":
            self._write_json(
                os.path.join(script_run, "quality_report_initial.json"),
                {
                    "status": "failed",
                    "pass": False,
                    "quality_report_phase": "initial",
                    "quality_stage_started": True,
                    "quality_stage_finished": False,
                    "quality_stage_interrupted": True,
                },
            )

        with open(script_path, "w", encoding="utf-8") as f:
            json.dump({"lines": []}, f)
        with open(source_path, "w", encoding="utf-8") as f:
            f.write("fuente extensa para reproducir escenarios de bundle\n")
        return script_ckpt, audio_ckpt, script_path, run_episode

    def test_replay_11_incidents_have_explicit_causal_mapping(self) -> None:
        allowed_statuses = {"found", "missing", "read_error", "not_applicable"}
        cases = [
            {"id": "2140", "scenario": "editorial_warn_only", "expected_action": "warn_only_non_blocking"},
            {"id": "2203", "scenario": "editorial_warn_only", "expected_action": "warn_only_non_blocking"},
            {"id": "2239", "scenario": "editorial_warn_only", "expected_action": "warn_only_non_blocking"},
            {"id": "2242", "scenario": "editorial_warn_only", "expected_action": "warn_only_non_blocking"},
            {"id": "2015", "scenario": "quality_interrupted", "expected_action": "fallback_to_initial_quality_report"},
            {"id": "2038", "scenario": "quality_interrupted", "expected_action": "fallback_to_initial_quality_report"},
            {"id": "2046", "scenario": "quality_interrupted", "expected_action": "fallback_to_initial_quality_report"},
            {"id": "2106", "scenario": "quality_interrupted", "expected_action": "fallback_to_initial_quality_report"},
            {"id": "2112", "scenario": "quality_interrupted", "expected_action": "fallback_to_initial_quality_report"},
            {"id": "2208", "scenario": "openai_empty_output_failed", "expected_action": "preserve_openai_empty_output_failure_kind"},
            {
                "id": "2120",
                "scenario": "root_mismatch",
                "mismatch": True,
                "expected_action": "autodiscovery_with_consistency_warning",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                with self.subTest(bundle_id=case["id"], action=case["expected_action"]):
                    case_root = os.path.join(tmp, case["id"])
                    os.makedirs(case_root, exist_ok=True)
                    script_ckpt, audio_ckpt, script_path, run_episode = self._build_case(
                        root=case_root,
                        bundle_id=case["id"],
                        scenario=str(case["scenario"]),
                        force_episode_mismatch=bool(case.get("mismatch", False)),
                    )
                    output_zip = os.path.join(case_root, "bundle.zip")
                    args = parse_args(
                        [
                            case["id"],
                            "--script-checkpoint-dir",
                            script_ckpt,
                            "--audio-checkpoint-dir",
                            audio_ckpt,
                            "--script-path",
                            script_path,
                            "--output",
                            output_zip,
                        ]
                    )
                    create_debug_bundle(args)
                    with zipfile.ZipFile(output_zip, "r") as zf:
                        names = set(zf.namelist())
                        self.assertIn("debug_bundle_metadata.json", names)
                        self.assertIn("collection_report.json", names)
                        metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                        collection = json.loads(zf.read("collection_report.json").decode("utf-8"))
                    self.assertEqual(metadata.get("bundle_version"), 2)
                    self.assertEqual(metadata.get("collection_report_path"), "collection_report.json")
                    statuses = {str(item.get("status", "")) for item in collection}
                    self.assertTrue(statuses.issubset(allowed_statuses))
                    self.assertEqual(int(metadata.get("collection_status_counts", {}).get("missing", 0)), 0)
                    self.assertTrue(bool(metadata.get("collection_complete", False)))
                    self.assertTrue(
                        all(
                            str(item.get("reason", "")).strip()
                            for item in collection
                            if str(item.get("status", "")).strip().lower() in {"missing", "not_applicable"}
                        )
                    )

                    scenario = str(case["scenario"])
                    if scenario == "editorial_warn_only":
                        quality_entries = [
                            item
                            for item in collection
                            if str(item.get("archive_name", "")).endswith("quality_report.json")
                            and str(item.get("status", "")).strip().lower() == "found"
                        ]
                        self.assertGreaterEqual(len(quality_entries), 1)
                        with zipfile.ZipFile(output_zip, "r") as zf:
                            report_payload = json.loads(
                                zf.read(str(quality_entries[0].get("archive_name", ""))).decode("utf-8")
                            )
                        self.assertTrue(bool(report_payload.get("editorial_warn_only", False)))
                        self.assertFalse(bool(report_payload.get("hard_fail_eligible", True)))
                    elif scenario == "quality_interrupted":
                        self.assertTrue(
                            any(
                                str(item.get("status", "")).strip().lower() == "found"
                                and str(item.get("reason", "")) == "initial_only_due_to_interruption"
                                and str(item.get("archive_name", "")).endswith("quality_report_initial.json")
                                for item in collection
                            )
                        )
                        self.assertFalse(
                            any(
                                str(item.get("status", "")).strip().lower() == "missing"
                                and str(item.get("archive_name", "")).endswith("quality_report.json")
                                for item in collection
                            )
                        )
                    elif scenario == "openai_empty_output_failed":
                        script_run_summary_entries = [
                            item
                            for item in collection
                            if str(item.get("category", "")) == "script_checkpoint"
                            and str(item.get("archive_name", "")).endswith("run_summary.json")
                            and str(item.get("status", "")).strip().lower() == "found"
                        ]
                        self.assertGreaterEqual(len(script_run_summary_entries), 1)
                        with zipfile.ZipFile(output_zip, "r") as zf:
                            run_summary_payload = json.loads(
                                zf.read(str(script_run_summary_entries[0].get("archive_name", ""))).decode("utf-8")
                            )
                        self.assertEqual(str(run_summary_payload.get("failure_kind", "")), "openai_empty_output")
                        self.assertTrue(
                            any(
                                str(item.get("status", "")).strip().lower() == "not_applicable"
                                and str(item.get("reason", "")) == "script_quality_gate_not_executed"
                                and str(item.get("archive_name", "")).endswith("quality_report.json")
                                for item in collection
                            )
                        )
                    elif scenario == "root_mismatch":
                        warnings = list(metadata.get("consistency_warnings", []))
                        self.assertGreaterEqual(len(warnings), 1)
                        self.assertEqual(str(metadata.get("resolved_episode_id", "")), run_episode)


if __name__ == "__main__":
    unittest.main()

