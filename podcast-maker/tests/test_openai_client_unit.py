import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig, ReliabilityConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.openai_client import OpenAIClient, _extract_text_from_responses_payload  # noqa: E402


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
        return False


def _http_error(code: int, body: str = "error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.example.com",
        code=code,
        msg="boom",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


class OpenAIClientUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        reliability = ReliabilityConfig.from_env()
        reliability = dataclasses.replace(
            reliability,
            max_requests_per_run=0,
            max_estimated_cost_usd=0.0,
        )
        self.client = OpenAIClient(
            api_key="test-key",
            logger=logger,
            reliability=reliability,
            script_model="gpt-test",
            script_reasoning_effort="low",
            tts_model="tts-test",
            script_timeout_seconds=5,
            script_retries=2,
            tts_timeout_seconds=5,
            tts_retries=2,
            tts_backoff_base_ms=1,
            tts_backoff_max_ms=2,
            circuit_breaker_failures=2,
        )

    def test_extract_text_prefers_output_blocks(self) -> None:
        payload = {
            "output": [{"content": [{"type": "output_text", "text": "hola "}, {"type": "x", "text": "bad"}]}]
        }
        self.assertEqual(_extract_text_from_responses_payload(payload), "hola")

    def test_extract_text_falls_back_to_output_text(self) -> None:
        payload = {"output_text": "fallback"}
        self.assertEqual(_extract_text_from_responses_payload(payload), "fallback")

    def test_check_budget_blocks_request_limit(self) -> None:
        self.client.reliability = dataclasses.replace(self.client.reliability, max_requests_per_run=1)
        self.client.requests_made = 1
        with self.assertRaises(RuntimeError):
            self.client._check_budget()

    def test_check_budget_blocks_cost_limit(self) -> None:
        self.client.reliability = dataclasses.replace(self.client.reliability, max_estimated_cost_usd=0.01)
        self.client.estimated_cost_usd = 0.02
        with self.assertRaises(RuntimeError):
            self.client._check_budget()

    def test_track_usage_increments_requests_and_cost(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ESTIMATED_COST_PER_SCRIPT_REQUEST_USD": "0.5", "ESTIMATED_COST_PER_TTS_REQUEST_USD": "0.25"},
            clear=False,
        ):
            self.client._track_usage(request_kind="script")
            self.client._track_usage(request_kind="tts")
        self.assertEqual(self.client.requests_made, 2)
        self.assertEqual(self.client.script_requests_made, 1)
        self.assertEqual(self.client.tts_requests_made, 1)
        self.assertAlmostEqual(self.client.estimated_cost_usd, 0.75, places=6)

    def test_track_usage_invalid_cost_env_falls_back_to_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ESTIMATED_COST_PER_SCRIPT_REQUEST_USD": "not-a-number",
                "ESTIMATED_COST_PER_TTS_REQUEST_USD": "bad",
            },
            clear=False,
        ):
            self.client._track_usage(request_kind="script")
            self.client._track_usage(request_kind="tts")
        self.assertEqual(self.client.requests_made, 2)
        self.assertAlmostEqual(self.client.estimated_cost_usd, 0.03, places=6)

    def test_track_usage_non_finite_cost_env_falls_back_to_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ESTIMATED_COST_PER_SCRIPT_REQUEST_USD": "nan",
                "ESTIMATED_COST_PER_TTS_REQUEST_USD": "inf",
            },
            clear=False,
        ):
            self.client._track_usage(request_kind="script")
            self.client._track_usage(request_kind="tts")
        self.assertFalse(math.isnan(self.client.estimated_cost_usd))
        self.assertAlmostEqual(self.client.estimated_cost_usd, 0.03, places=6)

    def test_track_usage_negative_cost_env_does_not_reduce_estimated_cost(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ESTIMATED_COST_PER_SCRIPT_REQUEST_USD": "-1",
                "ESTIMATED_COST_PER_TTS_REQUEST_USD": "-2",
            },
            clear=False,
        ):
            self.client._track_usage(request_kind="script")
            self.client._track_usage(request_kind="tts")
        self.assertEqual(self.client.requests_made, 2)
        self.assertEqual(self.client.estimated_cost_usd, 0.0)

    def test_post_json_script_circuit_breaker(self) -> None:
        self.client.circuit_breaker_failures = 1
        self.client.consecutive_script_failures = 1
        with self.assertRaises(RuntimeError):
            self.client._post_json(
                endpoint="https://api.openai.com/v1/responses",
                payload={},
                timeout_seconds=1,
                retries=1,
                request_kind="script",
                stage="s",
            )

    def test_post_json_retries_then_succeeds(self) -> None:
        responses = [
            urllib.error.URLError("timeout"),
            _Resp(json.dumps({"ok": True}).encode("utf-8")),
        ]

        def fake_urlopen(req, timeout):  # noqa: ANN001
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=fake_urlopen):
            with mock.patch("pipeline.openai_client.time.sleep"):
                with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                    payload = self.client._post_json(
                        endpoint="https://api.openai.com/v1/responses",
                        payload={"x": 1},
                        timeout_seconds=1,
                        retries=2,
                        request_kind="script",
                        stage="stage",
                    )
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(self.client.script_retries_total, 1)
        self.assertEqual(self.client.requests_made, 2)
        self.assertEqual(self.client.script_requests_made, 2)

    def test_post_json_failure_preserves_original_cause(self) -> None:
        err = urllib.error.URLError("network down")
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=err):
            with mock.patch("pipeline.openai_client.time.sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    self.client._post_json(
                        endpoint="https://api.openai.com/v1/responses",
                        payload={"x": 1},
                        timeout_seconds=1,
                        retries=1,
                        request_kind="script",
                        stage="stage_fail",
                    )
        self.assertIsInstance(ctx.exception.__cause__, urllib.error.URLError)

    def test_generate_script_json_parse_repair_success(self) -> None:
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        bad = {"output_text": "{bad-json"}
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="st",
                )
        self.assertIn("lines", payload)
        self.assertEqual(self.client.script_json_parse_failures, 1)
        self.assertEqual(self.client.script_json_parse_repair_successes, 1)

    def test_generate_script_json_extracts_wrapped_json_without_repair(self) -> None:
        wrapped = {
            "output_text": (
                "```json\n"
                '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
                "\n```"
            )
        }
        with mock.patch.object(self.client, "_post_json", return_value=wrapped):
            payload = self.client.generate_script_json(
                prompt="p",
                schema={"type": "object"},
                max_output_tokens=100,
                stage="wrapped_stage",
            )
        self.assertIn("lines", payload)
        self.assertEqual(self.client.script_json_parse_failures, 0)
        self.assertEqual(self.client.script_json_parse_repair_successes, 0)

    def test_generate_script_json_tracks_parse_failure_kind_and_repair_kind(self) -> None:
        bad = {"output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"'}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="kind_stage",
                )
        self.assertIn("lines", payload)
        self.assertEqual(int(self.client.script_json_parse_failures_by_kind.get("truncation", 0)), 1)
        self.assertEqual(int(self.client.script_json_parse_repair_successes_by_kind.get("truncation", 0)), 1)

    def test_generate_script_json_repair_parses_wrapped_repair_output(self) -> None:
        bad = {"output_text": "{bad-json"}
        wrapped_good = {
            "output_text": (
                "response:\n"
                '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}\n'
                "end"
            )
        }
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, wrapped_good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="wrapped_repair",
                )
        self.assertIn("lines", payload)
        self.assertEqual(self.client.script_json_parse_repair_successes, 1)

    def test_generate_script_json_invalid_repair_attempts_env_falls_back(self) -> None:
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        bad = {"output_text": "{bad-json"}
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "abc",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="st",
                )
        self.assertIn("lines", payload)
        self.assertEqual(self.client.script_json_parse_failures, 1)
        self.assertEqual(self.client.script_json_parse_repair_successes, 1)

    def test_generate_script_json_parse_repair_failure_raises(self) -> None:
        bad = {"output_text": "{bad-json"}
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, bad]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                with self.assertRaises(RuntimeError):
                    self.client.generate_script_json(
                        prompt="p",
                        schema={"type": "object"},
                        max_output_tokens=100,
                        stage="st",
                    )
        self.assertEqual(self.client.script_json_parse_failures, 1)
        self.assertEqual(self.client.script_json_parse_repair_successes, 0)
        self.assertGreaterEqual(self.client.script_json_parse_repair_failures, 1)
        self.assertGreaterEqual(int(self.client.script_json_parse_repair_failures_by_kind.get("truncation", 0)), 1)

    def test_generate_script_json_malformed_parse_failure_kind_and_repair_metrics(self) -> None:
        bad = {"output_text": '{"lines": bad}'}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                    "SCRIPT_PARSE_REPAIR_WRAPPER_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="malformed_stage",
                )
        self.assertIn("lines", payload)
        self.assertEqual(int(self.client.script_json_parse_failures_by_kind.get("malformed", 0)), 1)
        self.assertEqual(int(self.client.script_json_parse_repair_successes_by_kind.get("malformed", 0)), 1)

    def test_generate_script_json_truncation_adds_repair_attempts_and_scales_budget(self) -> None:
        bad_initial = {"output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"'}
        bad_repair = {"output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x"'}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        calls: list[dict[str, object]] = []

        def fake_post_json(*, endpoint, payload, timeout_seconds, retries, request_kind, stage):  # noqa: ANN001
            calls.append({"stage": stage, "max_output_tokens": payload.get("max_output_tokens")})
            if stage == "st":
                return bad_initial
            if stage == "st_parse_repair_1":
                return bad_repair
            if stage == "st_parse_repair_2":
                return good
            raise AssertionError(f"unexpected stage: {stage}")

        with mock.patch.object(self.client, "_post_json", side_effect=fake_post_json):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "2",
                    "SCRIPT_PARSE_REPAIR_OUTPUT_TOKENS_GROWTH": "2.0",
                    "SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS": "500",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="st",
                )

        self.assertIn("lines", payload)
        stages = [str(item.get("stage")) for item in calls]
        self.assertEqual(stages, ["st", "st_parse_repair_1", "st_parse_repair_2"])
        budget_1 = int(calls[1]["max_output_tokens"])  # type: ignore[arg-type]
        budget_2 = int(calls[2]["max_output_tokens"])  # type: ignore[arg-type]
        self.assertGreaterEqual(budget_1, 100)
        self.assertGreater(budget_2, budget_1)

    def test_generate_script_json_non_truncation_does_not_add_bonus_attempts(self) -> None:
        bad_non_truncated = {"output_text": '{"lines": bad}'}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        calls: list[str] = []

        def fake_post_json(*, endpoint, payload, timeout_seconds, retries, request_kind, stage):  # noqa: ANN001
            calls.append(str(stage))
            if stage == "st":
                return bad_non_truncated
            if stage == "st_parse_repair_1":
                return good
            raise AssertionError(f"unexpected stage: {stage}")

        with mock.patch.object(self.client, "_post_json", side_effect=fake_post_json):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "3",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="st",
                )

        self.assertIn("lines", payload)
        self.assertEqual(calls, ["st", "st_parse_repair_1"])

    def test_generate_script_json_wrapper_failures_use_wrapper_bonus_and_budget(self) -> None:
        bad_wrapper = {"output_text": 'prefix {bad-json} suffix'}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        calls: list[dict[str, object]] = []

        def fake_post_json(*, endpoint, payload, timeout_seconds, retries, request_kind, stage):  # noqa: ANN001
            calls.append({"stage": stage, "max_output_tokens": payload.get("max_output_tokens")})
            if stage == "st":
                return bad_wrapper
            if stage == "st_parse_repair_1":
                return bad_wrapper
            if stage == "st_parse_repair_2":
                return good
            raise AssertionError(f"unexpected stage: {stage}")

        with mock.patch.object(self.client, "_post_json", side_effect=fake_post_json):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_WRAPPER_BONUS_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_WRAPPER_OUTPUT_TOKENS_GROWTH": "1.6",
                    "SCRIPT_PARSE_REPAIR_WRAPPER_MAX_OUTPUT_TOKENS": "400",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="st",
                )

        self.assertIn("lines", payload)
        stages = [str(item.get("stage")) for item in calls]
        self.assertEqual(stages, ["st", "st_parse_repair_1", "st_parse_repair_2"])
        budget_1 = int(calls[1]["max_output_tokens"])  # type: ignore[arg-type]
        budget_2 = int(calls[2]["max_output_tokens"])  # type: ignore[arg-type]
        self.assertGreaterEqual(budget_1, 100)
        self.assertGreater(budget_2, budget_1)
        self.assertEqual(int(self.client.script_json_parse_failures_by_kind.get("wrapper", 0)), 1)

    def test_generate_script_json_tracks_stage_parse_and_repair_counters(self) -> None:
        bad = {"output_text": "{bad-json"}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        with mock.patch.object(self.client, "_post_json", side_effect=[bad, good]):
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_PARSE_REPAIR_ATTEMPTS": "1",
                    "SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS": "0",
                },
                clear=False,
            ):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="chunk_3",
                )
        self.assertIn("lines", payload)
        self.assertEqual(int(self.client.script_json_parse_failures_by_stage.get("chunk_3", 0)), 1)
        self.assertEqual(int(self.client.script_json_parse_repair_successes_by_stage.get("chunk_3", 0)), 1)

    def test_generate_script_json_empty_output_retries_then_succeeds(self) -> None:
        empty = {"output_text": ""}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }
        calls: list[str] = []

        def fake_post_json(*, endpoint, payload, timeout_seconds, retries, request_kind, stage):  # noqa: ANN001
            calls.append(str(stage))
            if stage == "empty_stage":
                return empty
            if stage == "empty_stage_empty_output_retry_1":
                return good
            raise AssertionError(f"unexpected stage: {stage}")

        with mock.patch.object(self.client, "_post_json", side_effect=fake_post_json):
            with mock.patch.dict(os.environ, {"SCRIPT_EMPTY_OUTPUT_RETRIES": "2"}, clear=False):
                payload = self.client.generate_script_json(
                    prompt="p",
                    schema={"type": "object"},
                    max_output_tokens=100,
                    stage="empty_stage",
                )
        self.assertIn("lines", payload)
        self.assertEqual(calls, ["empty_stage", "empty_stage_empty_output_retry_1"])
        self.assertEqual(int(self.client.script_empty_output_events), 1)
        self.assertEqual(int(self.client.script_empty_output_retries), 1)
        self.assertEqual(int(self.client.script_empty_output_failures), 0)
        self.assertEqual(int(self.client.script_json_parse_failures_by_kind.get("empty_output", 0)), 1)

    def test_generate_script_json_empty_output_exhausted_raises_openai_empty_output_marker(self) -> None:
        empty = {"output_text": ""}
        with mock.patch.object(self.client, "_post_json", side_effect=[empty, empty, empty]):
            with mock.patch.dict(os.environ, {"SCRIPT_EMPTY_OUTPUT_RETRIES": "2"}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    self.client.generate_script_json(
                        prompt="p",
                        schema={"type": "object"},
                        max_output_tokens=100,
                        stage="empty_fail_stage",
                    )
        self.assertIn("parse_failure_kind=empty_output", str(ctx.exception))
        self.assertEqual(int(self.client.script_empty_output_events), 1)
        self.assertEqual(int(self.client.script_empty_output_retries), 2)
        self.assertEqual(int(self.client.script_empty_output_failures), 1)
        self.assertEqual(int(self.client.script_json_parse_failures_by_kind.get("empty_output", 0)), 1)

    def test_generate_script_json_uses_timeout_override(self) -> None:
        observed: dict[str, int] = {}
        good = {
            "output_text": '{"lines":[{"speaker":"Ana","role":"Host1","instructions":"x","text":"hola"}]}'
        }

        def fake_post_json(*, endpoint, payload, timeout_seconds, retries, request_kind, stage):  # noqa: ANN001
            observed["timeout"] = int(timeout_seconds)
            return good

        with mock.patch.object(self.client, "_post_json", side_effect=fake_post_json):
            payload = self.client.generate_script_json(
                prompt="p",
                schema={"type": "object"},
                max_output_tokens=100,
                stage="override_timeout_stage",
                timeout_seconds_override=7,
            )
        self.assertIn("lines", payload)
        self.assertEqual(observed.get("timeout"), 7)

    def test_synthesize_speech_tts_circuit_breaker(self) -> None:
        self.client.circuit_breaker_failures = 1
        self.client.consecutive_tts_failures = 1
        with self.assertRaises(RuntimeError):
            self.client.synthesize_speech(
                text="hola",
                instructions="x",
                voice="cedar",
                stage="tts",
            )

    def test_synthesize_speech_uses_timeout_override(self) -> None:
        observed = {"timeout": None}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            observed["timeout"] = timeout
            return _Resp(b"ID3")

        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=fake_urlopen):
            audio = self.client.synthesize_speech(
                text="hola",
                instructions="x",
                voice="cedar",
                stage="tts",
                timeout_seconds_override=2,
            )
        self.assertEqual(audio, b"ID3")
        self.assertEqual(observed["timeout"], 2)

    def test_synthesize_speech_includes_speed_in_payload(self) -> None:
        observed: dict[str, object] = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            observed["timeout"] = timeout
            observed["payload"] = json.loads((req.data or b"{}").decode("utf-8"))
            return _Resp(b"ID3")

        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=fake_urlopen):
            audio = self.client.synthesize_speech(
                text="hola",
                instructions="x",
                voice="cedar",
                speed=1.23,
                stage="tts",
            )
        self.assertEqual(audio, b"ID3")
        payload = observed.get("payload", {})
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("speed"), 1.23)

    def test_synthesize_speech_clamps_speed_in_payload(self) -> None:
        observed: dict[str, object] = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            observed["payload"] = json.loads((req.data or b"{}").decode("utf-8"))
            return _Resp(b"ID3")

        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=fake_urlopen):
            self.client.synthesize_speech(
                text="hola",
                instructions="x",
                voice="cedar",
                speed=9.0,
                stage="tts",
            )
        payload = observed.get("payload", {})
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("speed"), 4.0)

    def test_synthesize_speech_retries_and_counts(self) -> None:
        sequence = [_http_error(429), _Resp(b"ID3")]

        def fake_urlopen(req, timeout):  # noqa: ANN001
            item = sequence.pop(0)
            if isinstance(item, urllib.error.HTTPError):
                raise item
            return item

        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=fake_urlopen):
            with mock.patch("pipeline.openai_client.time.sleep"):
                with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                    audio = self.client.synthesize_speech(
                        text="hola",
                        instructions="x",
                        voice="cedar",
                        stage="tts",
                    )
        self.assertEqual(audio, b"ID3")
        self.assertEqual(self.client.tts_retries_total, 1)
        self.assertEqual(self.client.requests_made, 2)
        self.assertEqual(self.client.tts_requests_made, 2)

    def test_synthesize_speech_5xx_uses_more_aggressive_backoff(self) -> None:
        sequence = [_http_error(502), _Resp(b"ID3")]
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=sequence):
            with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                with mock.patch("pipeline.openai_client.time.sleep") as sleep_spy:
                    audio = self.client.synthesize_speech(
                        text="hola",
                        instructions="x",
                        voice="cedar",
                        stage="tts_5xx",
                    )
        self.assertEqual(audio, b"ID3")
        sleep_spy.assert_called_once()
        waited = float(sleep_spy.call_args[0][0])
        self.assertGreater(waited, 0.001)

    def test_synthesize_speech_429_keeps_base_backoff(self) -> None:
        sequence = [_http_error(429), _Resp(b"ID3")]
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=sequence):
            with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                with mock.patch("pipeline.openai_client.time.sleep") as sleep_spy:
                    audio = self.client.synthesize_speech(
                        text="hola",
                        instructions="x",
                        voice="cedar",
                        stage="tts_429",
                    )
        self.assertEqual(audio, b"ID3")
        sleep_spy.assert_called_once()
        waited = float(sleep_spy.call_args[0][0])
        self.assertAlmostEqual(waited, 0.001, places=6)

    def test_post_json_5xx_uses_more_aggressive_backoff(self) -> None:
        sequence = [_http_error(502), _Resp(json.dumps({"ok": True}).encode("utf-8"))]
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=sequence):
            with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                with mock.patch("pipeline.openai_client.time.sleep") as sleep_spy:
                    self.client._post_json(
                        endpoint="https://api.openai.com/v1/responses",
                        payload={"x": 1},
                        timeout_seconds=1,
                        retries=2,
                        request_kind="script",
                        stage="stage_5xx",
                    )
        sleep_spy.assert_called_once()
        waited = float(sleep_spy.call_args[0][0])
        # base=0.001s in test client, 5xx multiplier should increase it.
        self.assertGreater(waited, 0.001)

    def test_post_json_429_keeps_base_backoff(self) -> None:
        sequence = [_http_error(429), _Resp(json.dumps({"ok": True}).encode("utf-8"))]
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=sequence):
            with mock.patch("pipeline.openai_client.random.uniform", return_value=0.0):
                with mock.patch("pipeline.openai_client.time.sleep") as sleep_spy:
                    self.client._post_json(
                        endpoint="https://api.openai.com/v1/responses",
                        payload={"x": 1},
                        timeout_seconds=1,
                        retries=2,
                        request_kind="script",
                        stage="stage_429",
                    )
        sleep_spy.assert_called_once()
        waited = float(sleep_spy.call_args[0][0])
        self.assertAlmostEqual(waited, 0.001, places=6)

    def test_synthesize_speech_failure_preserves_original_cause(self) -> None:
        err = urllib.error.URLError("network down")
        with mock.patch("pipeline.openai_client.urllib.request.urlopen", side_effect=err):
            with mock.patch("pipeline.openai_client.time.sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    self.client.synthesize_speech(
                        text="hola",
                        instructions="x",
                        voice="cedar",
                        stage="tts_fail",
                    )
        self.assertIsInstance(ctx.exception.__cause__, urllib.error.URLError)

    def test_from_configs_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self.client.logger
            reliability = self.client.reliability
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                with mock.patch(
                    "pipeline.openai_client.os.path.expanduser",
                    return_value=os.path.join(tmp, "missing_auth.json"),
                ):
                    with self.assertRaises(RuntimeError):
                        OpenAIClient.from_configs(
                            logger=logger,
                            reliability=reliability,
                            script_model="m",
                            tts_model="t",
                            script_timeout_seconds=10,
                            script_retries=1,
                            tts_timeout_seconds=10,
                            tts_retries=1,
                            tts_backoff_base_ms=100,
                            tts_backoff_max_ms=1000,
                        )

    def test_from_configs_uses_codex_auth_json_when_env_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = os.path.join(tmp, "auth.json")
            with open(auth_path, "w", encoding="utf-8") as f:
                json.dump({"OPENAI_API_KEY": "codex-key"}, f)
            logger = self.client.logger
            reliability = self.client.reliability
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                with mock.patch("pipeline.openai_client.os.path.expanduser", return_value=auth_path):
                    client = OpenAIClient.from_configs(
                        logger=logger,
                        reliability=reliability,
                        script_model="m",
                        tts_model="t",
                        script_timeout_seconds=10,
                        script_retries=1,
                        tts_timeout_seconds=10,
                        tts_retries=1,
                        tts_backoff_base_ms=100,
                        tts_backoff_max_ms=1000,
                    )
        self.assertEqual(client.api_key, "codex-key")

    def test_from_configs_prefers_env_over_codex_auth_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = os.path.join(tmp, "auth.json")
            with open(auth_path, "w", encoding="utf-8") as f:
                json.dump({"OPENAI_API_KEY": "codex-key"}, f)
            logger = self.client.logger
            reliability = self.client.reliability
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
                with mock.patch("pipeline.openai_client.os.path.expanduser", return_value=auth_path):
                    client = OpenAIClient.from_configs(
                        logger=logger,
                        reliability=reliability,
                        script_model="m",
                        tts_model="t",
                        script_timeout_seconds=10,
                        script_retries=1,
                        tts_timeout_seconds=10,
                        tts_retries=1,
                        tts_backoff_base_ms=100,
                        tts_backoff_max_ms=1000,
                    )
        self.assertEqual(client.api_key, "env-key")

    def test_from_configs_reasoning_effort_defaults_to_low(self) -> None:
        logger = self.client.logger
        reliability = self.client.reliability
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "SCRIPT_REASONING_EFFORT": ""},
            clear=False,
        ):
            client = OpenAIClient.from_configs(
                logger=logger,
                reliability=reliability,
                script_model="m",
                tts_model="t",
                script_timeout_seconds=10,
                script_retries=1,
                tts_timeout_seconds=10,
                tts_retries=1,
                tts_backoff_base_ms=100,
                tts_backoff_max_ms=1000,
            )
        self.assertEqual(client.script_reasoning_effort, "low")

    def test_from_configs_reasoning_effort_override(self) -> None:
        logger = self.client.logger
        reliability = self.client.reliability
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "SCRIPT_REASONING_EFFORT": "HIGH"},
            clear=False,
        ):
            client = OpenAIClient.from_configs(
                logger=logger,
                reliability=reliability,
                script_model="m",
                tts_model="t",
                script_timeout_seconds=10,
                script_retries=1,
                tts_timeout_seconds=10,
                tts_retries=1,
                tts_backoff_base_ms=100,
                tts_backoff_max_ms=1000,
            )
        self.assertEqual(client.script_reasoning_effort, "high")

    def test_from_configs_invalid_reasoning_effort_falls_back_to_low(self) -> None:
        logger = self.client.logger
        reliability = self.client.reliability
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "SCRIPT_REASONING_EFFORT": "turbo"},
            clear=False,
        ):
            client = OpenAIClient.from_configs(
                logger=logger,
                reliability=reliability,
                script_model="m",
                tts_model="t",
                script_timeout_seconds=10,
                script_retries=1,
                tts_timeout_seconds=10,
                tts_retries=1,
                tts_backoff_base_ms=100,
                tts_backoff_max_ms=1000,
            )
        self.assertEqual(client.script_reasoning_effort, "low")

    def test_script_json_payload_uses_configured_reasoning_effort(self) -> None:
        self.client.script_reasoning_effort = "high"
        payload = self.client._script_json_payload(
            prompt="p",
            schema={"type": "object"},
            max_output_tokens=100,
        )
        self.assertEqual(payload.get("reasoning", {}).get("effort"), "high")

    def test_generate_freeform_text_uses_high_reasoning_for_quality_eval_stage(self) -> None:
        self.client.script_reasoning_effort = "low"
        with mock.patch.object(self.client, "_post_json", return_value={"output_text": "ok"}) as post_mock:
            out = self.client.generate_freeform_text(
                prompt="evaluate",
                max_output_tokens=120,
                stage="script_quality_eval",
            )
        self.assertEqual(out, "ok")
        payload = dict(post_mock.call_args.kwargs.get("payload", {}))
        self.assertEqual(payload.get("reasoning", {}).get("effort"), "high")

    def test_generate_freeform_text_uses_default_reasoning_for_non_quality_stage(self) -> None:
        self.client.script_reasoning_effort = "medium"
        with mock.patch.object(self.client, "_post_json", return_value={"output_text": "ok"}) as post_mock:
            out = self.client.generate_freeform_text(
                prompt="helper",
                max_output_tokens=120,
                stage="script_quality_semantic_rules",
            )
        self.assertEqual(out, "ok")
        payload = dict(post_mock.call_args.kwargs.get("payload", {}))
        self.assertEqual(payload.get("reasoning", {}).get("effort"), "medium")

    def test_from_configs_invalid_circuit_breaker_env_falls_back_to_zero(self) -> None:
        logger = self.client.logger
        reliability = self.client.reliability
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "OPENAI_CIRCUIT_BREAKER_FAILURES": "abc"},
            clear=False,
        ):
            client = OpenAIClient.from_configs(
                logger=logger,
                reliability=reliability,
                script_model="m",
                tts_model="t",
                script_timeout_seconds=10,
                script_retries=1,
                tts_timeout_seconds=10,
                tts_retries=1,
                tts_backoff_base_ms=100,
                tts_backoff_max_ms=1000,
            )
        self.assertEqual(client.circuit_breaker_failures, 0)


if __name__ == "__main__":
    unittest.main()

