"""Regression tests distinguishing left-neighbor continuity from source fade-in."""

import math
import unittest
from unittest.mock import Mock, patch

import torch

from engine.timeline_session_manager import TimelineSession


class TestTimelineLeftSeam(unittest.TestCase):
    def make_session(self, completed=True, start=8):
        session = TimelineSession.__new__(TimelineSession)
        session.meta = {
            "total_frames": start + 8, "width": 16, "height": 16, "fps": 24,
            "source_video_path": "original.mp4",
            "chunks": [
                {"chunk_index": 0, "start_frame": 0, "end_frame": start,
                 "status": "completed" if completed else "unprocessed"},
                {"chunk_index": 1, "start_frame": start, "end_frame": start + 8, "status": "unprocessed"},
            ],
        }
        session.master_frames = torch.arange(start + 8).view(-1, 1, 1, 1).expand(-1, 16, 16, 3).float() / 20
        session.master_audio = None
        session.has_chunk_patch = Mock(return_value=False)
        session.save_chunk_patch = Mock()
        session._save_chunk_thumbnail = Mock()
        session._update_gap_analysis = Mock()
        session.save_to_disk = Mock()
        return session

    def apply_patch(self, session, start=8):
        session.patch_chunk(1, start, start + 8, torch.full((8, 16, 16, 3), 0.9), seam_blend_frames=4)
        return session.save_chunk_patch.call_args.args[1]

    def assert_blend(self, result, reference):
        alpha = (0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, 4))).view(4, 1, 1, 1)
        torch.testing.assert_close(result[:4], reference * (1 - alpha) + 0.9 * alpha)
        torch.testing.assert_close(result[4:], torch.full((4, 16, 16, 3), 0.9))

    def test_completed_master_fallback_matches_previous_patch(self):
        master_session = self.make_session()
        reference = master_session.master_frames[4:8].clone()
        previous = master_session.master_frames[:8].clone()
        result = self.apply_patch(master_session)
        self.assert_blend(result, reference)
        torch.testing.assert_close(master_session.master_frames[:8], previous)
        patch_session = self.make_session()
        patch_session.has_chunk_patch.return_value = True
        patch_session.load_chunk_patch = Mock(return_value={"frames": previous})
        torch.testing.assert_close(self.apply_patch(patch_session), result)

    def test_short_patch_falls_back_to_previous_master_tail(self):
        session = self.make_session()
        reference = session.master_frames[4:8].clone()
        session.has_chunk_patch.return_value = True
        session.load_chunk_patch = Mock(return_value={"frames": torch.zeros((2, 16, 16, 3))})
        self.assert_blend(self.apply_patch(session), reference)

    def test_unedited_predecessor_uses_current_original_head(self):
        session = self.make_session(completed=False)
        reference = session.master_frames[8:12].clone()
        self.assert_blend(self.apply_patch(session), reference)

    def test_unedited_predecessor_can_use_source_video(self):
        session = self.make_session(completed=False)
        session.master_frames = None
        reference = torch.full((4, 16, 16, 3), 0.3)
        with patch("engine.timeline_session_manager.os.path.isfile", return_value=True), patch(
            "engine.timeline_session_manager.extract_video_chunk_ffmpeg", return_value=reference
        ) as extract:
            self.assert_blend(self.apply_patch(session), reference)
            self.assertEqual(extract.call_args.kwargs["start_frame"], 8)

    def test_missing_completed_edit_does_not_use_original_video(self):
        session = self.make_session()
        session.master_frames = None
        with patch("engine.timeline_session_manager.extract_video_chunk_ffmpeg") as extract:
            torch.testing.assert_close(self.apply_patch(session), torch.full((8, 16, 16, 3), 0.9))
            extract.assert_not_called()

    def test_insufficient_preceding_frames_skip_blending(self):
        session = self.make_session(start=2)
        torch.testing.assert_close(self.apply_patch(session, start=2), torch.full((8, 16, 16, 3), 0.9))


if __name__ == "__main__":
    unittest.main()
