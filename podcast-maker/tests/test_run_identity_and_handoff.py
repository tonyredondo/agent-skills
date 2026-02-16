import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_podcast  # noqa: E402


class RunIdentityAndHandoffTests(unittest.TestCase):
    def test_run_podcast_passes_same_episode_id_to_script_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 80)
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)

            with mock.patch.object(run_podcast.make_script, "main", return_value=0) as script_main:
                with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                    rc = run_podcast.main(
                        [
                            source,
                            outdir,
                            "show_15min",
                            "--episode-id",
                            "episode_shared",
                            "--profile",
                            "standard",
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(script_main.call_count, 1)
            self.assertEqual(podcast_main.call_count, 1)
            script_argv = list(script_main.call_args.args[0])
            podcast_argv = list(podcast_main.call_args.args[0])
            self.assertIn("--episode-id", script_argv)
            self.assertIn("--episode-id", podcast_argv)
            self.assertIn("--run-token", script_argv)
            self.assertIn("--run-token", podcast_argv)
            self.assertEqual(script_argv[script_argv.index("--episode-id") + 1], "episode_shared")
            self.assertEqual(podcast_argv[podcast_argv.index("--episode-id") + 1], "episode_shared")
            self.assertEqual(
                script_argv[script_argv.index("--run-token") + 1],
                podcast_argv[podcast_argv.index("--run-token") + 1],
            )
            self.assertIn(os.path.join(outdir, "episode_shared_script.json"), script_argv)

    def test_run_podcast_does_not_start_audio_when_script_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 60)
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)

            with mock.patch.object(run_podcast.make_script, "main", return_value=2) as script_main:
                with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                    rc = run_podcast.main([source, outdir, "show_15min"])

            self.assertEqual(rc, 2)
            self.assertEqual(script_main.call_count, 1)
            self.assertEqual(podcast_main.call_count, 0)

    def test_run_podcast_retries_script_stage_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 60)
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)

            with mock.patch.dict(os.environ, {"RUN_PODCAST_SCRIPT_ATTEMPTS": "2"}, clear=False):
                with mock.patch.object(run_podcast.make_script, "main", side_effect=[1, 0]) as script_main:
                    with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                        rc = run_podcast.main([source, outdir, "show_15min"])

            self.assertEqual(rc, 0)
            self.assertEqual(script_main.call_count, 2)
            self.assertEqual(podcast_main.call_count, 1)
            second_call_argv = list(script_main.call_args_list[1].args[0])
            self.assertIn("--resume", second_call_argv)
            self.assertIn("--resume-force", second_call_argv)
            self.assertIn("--force-unlock", second_call_argv)

    def test_run_podcast_sets_default_audio_checkpoint_dir_for_both_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 60)
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            expected_audio_ckpt = os.path.join(outdir, ".audio_checkpoints")
            observed: dict[str, str] = {}

            def _script_main(_argv):  # noqa: ANN001, ANN202
                observed["script"] = str(os.environ.get("AUDIO_CHECKPOINT_DIR", ""))
                return 0

            def _podcast_main(_argv):  # noqa: ANN001, ANN202
                observed["audio"] = str(os.environ.get("AUDIO_CHECKPOINT_DIR", ""))
                return 0

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AUDIO_CHECKPOINT_DIR", None)
                with mock.patch.object(run_podcast.make_script, "main", side_effect=_script_main):
                    with mock.patch.object(run_podcast.make_podcast, "main", side_effect=_podcast_main):
                        rc = run_podcast.main([source, outdir, "show_15min"])
                self.assertNotIn("AUDIO_CHECKPOINT_DIR", os.environ)

            self.assertEqual(rc, 0)
            self.assertEqual(observed.get("script"), expected_audio_ckpt)
            self.assertEqual(observed.get("audio"), expected_audio_ckpt)

    def test_run_podcast_respects_preconfigured_audio_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 60)
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            preset_audio_ckpt = os.path.join(tmp, "shared_audio_ckpt")
            observed: dict[str, str] = {}

            def _script_main(_argv):  # noqa: ANN001, ANN202
                observed["script"] = str(os.environ.get("AUDIO_CHECKPOINT_DIR", ""))
                return 0

            def _podcast_main(_argv):  # noqa: ANN001, ANN202
                observed["audio"] = str(os.environ.get("AUDIO_CHECKPOINT_DIR", ""))
                return 0

            with mock.patch.dict(os.environ, {"AUDIO_CHECKPOINT_DIR": preset_audio_ckpt}, clear=False):
                with mock.patch.object(run_podcast.make_script, "main", side_effect=_script_main):
                    with mock.patch.object(run_podcast.make_podcast, "main", side_effect=_podcast_main):
                        rc = run_podcast.main([source, outdir, "show_15min"])
                self.assertEqual(os.environ.get("AUDIO_CHECKPOINT_DIR"), preset_audio_ckpt)

            self.assertEqual(rc, 0)
            self.assertEqual(observed.get("script"), preset_audio_ckpt)
            self.assertEqual(observed.get("audio"), preset_audio_ckpt)

    def test_run_podcast_does_not_retry_nonrecoverable_source_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 20)
            outdir = os.path.join(tmp, "out")
            script_ckpt = os.path.join(tmp, "script_ckpt")
            os.makedirs(outdir, exist_ok=True)
            run_summary_path = os.path.join(script_ckpt, "show_15min", "run_summary.json")

            def _script_main(_argv):  # noqa: ANN001, ANN202
                os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
                with open(run_summary_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump({"failure_kind": "source_too_short"}, f)
                return 1

            with mock.patch.dict(
                os.environ,
                {
                    "RUN_PODCAST_SCRIPT_ATTEMPTS": "3",
                    "SCRIPT_CHECKPOINT_DIR": script_ckpt,
                },
                clear=False,
            ):
                with mock.patch.object(run_podcast.make_script, "main", side_effect=_script_main) as script_main:
                    with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                        rc = run_podcast.main([source, outdir, "show_15min"])

            self.assertEqual(rc, 1)
            self.assertEqual(script_main.call_count, 1)
            self.assertEqual(podcast_main.call_count, 0)

    def test_run_podcast_does_not_retry_nonrecoverable_resume_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 20)
            outdir = os.path.join(tmp, "out")
            script_ckpt = os.path.join(tmp, "script_ckpt")
            os.makedirs(outdir, exist_ok=True)
            run_summary_path = os.path.join(script_ckpt, "show_15min", "run_summary.json")

            def _script_main(_argv):  # noqa: ANN001, ANN202
                os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
                with open(run_summary_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump({"failure_kind": "resume_blocked"}, f)
                return 1

            with mock.patch.dict(
                os.environ,
                {
                    "RUN_PODCAST_SCRIPT_ATTEMPTS": "3",
                    "SCRIPT_CHECKPOINT_DIR": script_ckpt,
                },
                clear=False,
            ):
                with mock.patch.object(run_podcast.make_script, "main", side_effect=_script_main) as script_main:
                    with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                        rc = run_podcast.main([source, outdir, "show_15min"])

            self.assertEqual(rc, 1)
            self.assertEqual(script_main.call_count, 1)
            self.assertEqual(podcast_main.call_count, 0)

    def test_run_podcast_ignores_stale_failure_kind_token_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido base " * 40)
            outdir = os.path.join(tmp, "out")
            script_ckpt = os.path.join(tmp, "script_ckpt")
            os.makedirs(outdir, exist_ok=True)
            run_summary_path = os.path.join(script_ckpt, "show_15min", "run_summary.json")
            calls = {"n": 0}

            def _script_main(_argv):  # noqa: ANN001, ANN202
                calls["n"] += 1
                if calls["n"] == 1:
                    os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
                    with open(run_summary_path, "w", encoding="utf-8") as f:
                        import json

                        json.dump(
                            {
                                "failure_kind": "source_too_short",
                                "run_token": "stale-run-token",
                            },
                            f,
                        )
                    return 1
                return 0

            with mock.patch.dict(
                os.environ,
                {
                    "RUN_PODCAST_SCRIPT_ATTEMPTS": "2",
                    "SCRIPT_CHECKPOINT_DIR": script_ckpt,
                },
                clear=False,
            ):
                with mock.patch.object(run_podcast.make_script, "main", side_effect=_script_main) as script_main:
                    with mock.patch.object(run_podcast.make_podcast, "main", return_value=0) as podcast_main:
                        rc = run_podcast.main([source, outdir, "show_15min"])

            self.assertEqual(rc, 0)
            self.assertEqual(script_main.call_count, 2)
            self.assertEqual(podcast_main.call_count, 1)


if __name__ == "__main__":
    unittest.main()

