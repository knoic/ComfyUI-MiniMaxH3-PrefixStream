"""Regression coverage for multichannel PCM export and mixed-resolution stitching."""

import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from engine.timeline_session_manager import export_master_to_video_file
from pipeline.seam_protector import stitch_video_images


class TestMultichannelExport(unittest.TestCase):
    def test_wav_preserves_sample_count_and_channel_order(self):
        for channels in (1, 2, 4, 6):
            with self.subTest(channels=channels), tempfile.TemporaryDirectory() as directory:
                samples = 120
                waveform = torch.arange(channels * samples).reshape(1, channels, samples).float() / (channels * samples)
                session = SimpleNamespace(
                    meta={"fps": 24}, master_frames=torch.zeros(2, 16, 16, 3),
                    master_audio={"waveform": waveform, "sample_rate": 48000},
                )
                def inspect_wav(cmd, **kwargs):
                    inputs = [cmd[i + 1] for i, arg in enumerate(cmd[:-1]) if arg == "-i"]
                    with wave.open(inputs[1], "rb") as audio:
                        self.assertEqual(audio.getnchannels(), channels)
                        self.assertEqual(audio.getnframes(), samples)
                        self.assertEqual(audio.getframerate(), 48000)
                        self.assertEqual(audio.getsampwidth(), 2)
                        actual = np.frombuffer(audio.readframes(samples), dtype="<i2").reshape(samples, channels)
                    expected = (waveform[0].numpy() * 32767).astype(np.int16).T
                    np.testing.assert_array_equal(actual, expected)
                    return Mock(returncode=0, communicate=Mock(return_value=(b"", b"")))
                with patch("engine.timeline_session_manager.get_or_create_timeline_session", return_value=session), patch(
                    "engine.timeline_session_manager.get_ffmpeg_path", return_value="ffmpeg"
                ), patch("engine.timeline_session_manager.subprocess.Popen", side_effect=inspect_wav) as popen:
                    result = export_master_to_video_file("test", output_dir=directory)
                    popen.assert_called_once()
                    self.assertTrue(result["has_audio"])


class TestMixedResolutionStitch(unittest.TestCase):
    def test_resize_overlap_and_body_before_blending(self):
        previous = torch.full((6, 16, 24, 3), 0.2, dtype=torch.float64)
        current = torch.linspace(0, 1, 8 * 20 * 30 * 3).reshape(8, 20, 30, 3)
        original = current.clone()
        resized = F.interpolate(current.permute(0, 3, 1, 2), size=(16, 24), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        for crossfade in (0, 2):
            for luminance in (False, True):
                with self.subTest(crossfade=crossfade, luminance=luminance):
                    result = stitch_video_images(previous, current, trim_frames=3, crossfade_frames=crossfade, luminance_match=luminance)
                    expected = stitch_video_images(previous, resized, trim_frames=3, crossfade_frames=crossfade, luminance_match=luminance)
                    self.assertEqual(result.shape, (11, 16, 24, 3))
                    self.assertEqual(result.dtype, previous.dtype)
                    torch.testing.assert_close(result, expected)
        torch.testing.assert_close(current, original)

    def test_no_previous_clip_keeps_current_resolution(self):
        current = torch.zeros(8, 20, 30, 3)
        result = stitch_video_images(None, current, trim_frames=3)
        self.assertEqual(result.shape, (5, 20, 30, 3))

    def test_channel_mismatch_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "channel counts"):
            stitch_video_images(torch.zeros(6, 16, 24, 3), torch.zeros(8, 20, 30, 4), trim_frames=3)


if __name__ == "__main__":
    unittest.main()
