import dataclasses
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
from typing import Optional


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.alibaba_tts_client import AlibabaInstructTTSProvider  # noqa: E402
from pipeline.config import LoggingConfig, ReliabilityConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402


class _Resp:
    def __init__(self, *, data: bytes, content_type: str = "application/json") -> None:
        self._data = data
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._data

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
        return False


class AlibabaTTSProviderTests(unittest.TestCase):
    def _provider(self, *, reliability: Optional[ReliabilityConfig] = None) -> AlibabaInstructTTSProvider:
        return AlibabaInstructTTSProvider(
            api_key="dashscope-secret-key",
            model_name="qwen3-tts-instruct-flash",
            base_url="https://dashscope-intl.aliyuncs.com/api/v1",
            language_type="Spanish",
            optimize_instructions=True,
            timeout_seconds=5,
            retries=2,
            backoff_base_ms=1,
            backoff_max_ms=2,
            reliability=reliability or ReliabilityConfig.from_env(),
            logger=Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            ),
        )

    def test_synthesize_speech_posts_payload_and_downloads_audio(self) -> None:
        provider = self._provider()
        captured_payload: dict[str, object] = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "multimodal-generation/generation" in url:
                captured_payload.update(json.loads((req.data or b"{}").decode("utf-8")))
                return _Resp(
                    data=json.dumps(
                        {"output": {"audio": {"url": "https://example.test/audio/result.wav"}}}
                    ).encode("utf-8")
                )
            return _Resp(data=b"RIFFxxxxWAVE", content_type="audio/wav")

        with unittest.mock.patch("pipeline.alibaba_tts_client.urllib.request.urlopen", side_effect=fake_urlopen):
            result = provider.synthesize_speech(
                text="Hola mundo",
                instructions="Speak with high enthusiasm",
                voice="Cherry",
                stage="tts_segment_0001",
            )
        self.assertEqual(result.file_extension, "wav")
        self.assertEqual(result.content_type, "audio/wav")
        self.assertEqual(result.provider, "alibaba")
        self.assertEqual(result.model, "qwen3-tts-instruct-flash")
        self.assertEqual(captured_payload.get("model"), "qwen3-tts-instruct-flash")
        input_payload = captured_payload.get("input")
        self.assertIsInstance(input_payload, dict)
        self.assertEqual(input_payload.get("voice"), "Cherry")  # type: ignore[union-attr]
        self.assertEqual(input_payload.get("language_type"), "Spanish")  # type: ignore[union-attr]

    def test_budget_guardrail_applies_max_requests(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), max_requests_per_run=1, max_estimated_cost_usd=0.0)
        provider = self._provider(reliability=reliability)

        def fake_urlopen(req, timeout):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "multimodal-generation/generation" in url:
                return _Resp(
                    data=json.dumps(
                        {"output": {"audio": {"url": "https://example.test/audio/result.wav"}}}
                    ).encode("utf-8")
                )
            return _Resp(data=b"RIFFxxxxWAVE", content_type="audio/wav")

        with unittest.mock.patch("pipeline.alibaba_tts_client.urllib.request.urlopen", side_effect=fake_urlopen):
            provider.synthesize_speech(
                text="uno",
                instructions="x",
                voice="Cherry",
                stage="s1",
            )
            with self.assertRaises(RuntimeError):
                provider.synthesize_speech(
                    text="dos",
                    instructions="x",
                    voice="Cherry",
                    stage="s2",
                )

    def test_reserve_request_slot_thread_safety(self) -> None:
        provider = self._provider(
            reliability=dataclasses.replace(
                ReliabilityConfig.from_env(),
                max_requests_per_run=0,
                max_estimated_cost_usd=0.0,
            )
        )

        def worker() -> None:
            for _ in range(50):
                provider.reserve_request_slot()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(provider.requests_made, 400)

    def test_failure_message_does_not_expose_api_key(self) -> None:
        provider = self._provider()
        with unittest.mock.patch(
            "pipeline.alibaba_tts_client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                provider.synthesize_speech(
                    text="hola",
                    instructions="x",
                    voice="Cherry",
                    stage="s",
                )
        self.assertNotIn("dashscope-secret-key", str(ctx.exception))

    def test_http_error_detail_log_redacts_api_key(self) -> None:
        provider = self._provider()
        fake_logger = unittest.mock.Mock()
        provider.logger = fake_logger  # type: ignore[assignment]
        http_error = urllib.error.HTTPError(
            url="https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            code=401,
            msg="unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"invalid key dashscope-secret-key"}'),
        )
        with unittest.mock.patch(
            "pipeline.alibaba_tts_client.urllib.request.urlopen",
            side_effect=http_error,
        ):
            with self.assertRaises(RuntimeError):
                provider.synthesize_speech(
                    text="hola",
                    instructions="x",
                    voice="Cherry",
                    stage="s",
                )
        self.assertTrue(fake_logger.warn.called)
        _, kwargs = fake_logger.warn.call_args
        self.assertNotIn("dashscope-secret-key", str(kwargs.get("detail", "")))

    def test_wrapped_error_redacts_api_key_text(self) -> None:
        provider = self._provider()
        with unittest.mock.patch(
            "pipeline.alibaba_tts_client.urllib.request.urlopen",
            side_effect=RuntimeError("auth failed for dashscope-secret-key"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                provider.synthesize_speech(
                    text="hola",
                    instructions="x",
                    voice="Cherry",
                    stage="s",
                )
        self.assertNotIn("dashscope-secret-key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
