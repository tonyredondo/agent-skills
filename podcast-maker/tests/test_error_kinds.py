import os
import socket
import sys
import unittest
import urllib.error


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.errors import (  # noqa: E402
    ERROR_KIND_INTERRUPTED,
    ERROR_KIND_NETWORK,
    ERROR_KIND_OPENAI_EMPTY_OUTPUT,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_SOURCE_TOO_SHORT,
    ERROR_KIND_STUCK,
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_UNKNOWN,
    ScriptOperationError,
    TTSBatchError,
    TTSOperationError,
    classify_tts_exception,
    is_stuck_error_kind,
)


class ErrorKindsTests(unittest.TestCase):
    def test_openai_empty_output_kind_constant(self) -> None:
        self.assertEqual(ERROR_KIND_OPENAI_EMPTY_OUTPUT, "openai_empty_output")

    def test_operation_error_preserves_kind(self) -> None:
        err = TTSOperationError("boom", error_kind=ERROR_KIND_RATE_LIMIT)
        self.assertEqual(classify_tts_exception(err), ERROR_KIND_RATE_LIMIT)

    def test_script_operation_error_preserves_kind(self) -> None:
        err = ScriptOperationError("source too short", error_kind=ERROR_KIND_SOURCE_TOO_SHORT)
        self.assertEqual(err.error_kind, ERROR_KIND_SOURCE_TOO_SHORT)

    def test_classify_runtime_messages(self) -> None:
        self.assertEqual(classify_tts_exception(RuntimeError("HTTP 429 rate limit")), ERROR_KIND_RATE_LIMIT)
        self.assertEqual(classify_tts_exception(RuntimeError("TTS global timeout reached")), ERROR_KIND_TIMEOUT)
        self.assertEqual(classify_tts_exception(RuntimeError("TTS stuck detected in chunk 1")), ERROR_KIND_STUCK)
        self.assertEqual(
            classify_tts_exception(RuntimeError("TTS request failed: <urlopen error timed out>")),
            ERROR_KIND_TIMEOUT,
        )
        self.assertEqual(
            classify_tts_exception(RuntimeError("Resume blocked: audio config fingerprint changed")),
            ERROR_KIND_RESUME_BLOCKED,
        )
        self.assertEqual(
            classify_tts_exception(RuntimeError("urlopen error [Errno -2] Name or service not known")),
            ERROR_KIND_NETWORK,
        )
        self.assertEqual(classify_tts_exception(RuntimeError("stream is now unstuck after retry")), ERROR_KIND_UNKNOWN)
        self.assertEqual(classify_tts_exception(RuntimeError("something else")), ERROR_KIND_UNKNOWN)

    def test_classify_uses_exception_types_in_chain(self) -> None:
        http_429 = urllib.error.HTTPError(
            url="https://api.example.com",
            code=429,
            msg="Too many requests",
            hdrs=None,
            fp=None,
        )
        wrapped_rate = RuntimeError("wrapped error")
        wrapped_rate.__cause__ = http_429
        self.assertEqual(classify_tts_exception(wrapped_rate), ERROR_KIND_RATE_LIMIT)

        timeout_cause = urllib.error.URLError(socket.timeout("timed out"))
        wrapped_timeout = RuntimeError("wrapped error")
        wrapped_timeout.__cause__ = timeout_cause
        self.assertEqual(classify_tts_exception(wrapped_timeout), ERROR_KIND_TIMEOUT)

        network_cause = urllib.error.URLError(OSError("temporary failure in name resolution"))
        wrapped_network = RuntimeError("wrapped error")
        wrapped_network.__cause__ = network_cause
        self.assertEqual(classify_tts_exception(wrapped_network), ERROR_KIND_NETWORK)

        interrupted = InterruptedError("ctrl-c")
        wrapped_interrupted = RuntimeError("wrapped interruption")
        wrapped_interrupted.__cause__ = interrupted
        self.assertEqual(classify_tts_exception(wrapped_interrupted), ERROR_KIND_INTERRUPTED)

    def test_batch_error_sets_stuck_abort_and_primary_kind(self) -> None:
        err = TTSBatchError(
            manifest_path="/tmp/manifest.json",
            failed_segments=[{"segment_id": "0001"}],
            failed_kinds=[ERROR_KIND_TIMEOUT, ERROR_KIND_TIMEOUT],
        )
        self.assertEqual(err.primary_kind, ERROR_KIND_TIMEOUT)
        self.assertEqual(err.failed_kinds, [ERROR_KIND_TIMEOUT])
        self.assertTrue(err.stuck_abort)
        self.assertIn("kinds=[timeout]", str(err))

    def test_stuck_kind_helper(self) -> None:
        self.assertTrue(is_stuck_error_kind(ERROR_KIND_TIMEOUT))
        self.assertTrue(is_stuck_error_kind(ERROR_KIND_STUCK))
        self.assertFalse(is_stuck_error_kind(ERROR_KIND_RATE_LIMIT))


if __name__ == "__main__":
    unittest.main()

