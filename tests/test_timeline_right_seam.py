"""Regression tests for right-seam patch and master-frame fallbacks."""

import math
import unittest
from unittest.mock import Mock

import torch

from engine.timeline_session_manager import TimelineSession


class TestTimelineRightSeam(unittest.TestCase):
    def make_session(self, master_length=12):
        session = TimelineSession.__new__(TimelineSession)
        session.meta = {
            "total_frames": 12, "width": 16, "height": 16, "fps": 24,
            "chunks": [
                {"chunk_index": 0, "start_frame": 0, "end_frame": 8, "status": "unprocessed"},
                {"chunk_index": 1, "start_frame": 8, "end_frame": 12, "status": "completed"},
            ],
        }
        session.master_frames = torch.full((master_length, 16, 16, 3), 0.1)
        session.master_frames[8:] = torch.arange(master_length - 8).view(-1, 1, 1, 1) * 0.1 + 0.2
        session.master_audio = None
        session.has_chunk_patch = Mock(return_value=False)
        session.save_chunk_patch = Mock()
        session._save_chunk_thumbnail = Mock()
        session._update_gap_analysis = Mock()
        session.save_to_disk = Mock()
        return session

    def apply_patch(self, session):
        session.patch_chunk(0, 0, 8, torch.full((8, 16, 16, 3), 0.9), seam_blend_frames=4)

    def assert_blend(self, session, right):
        alpha = (0.5 + 0.5 * torch.cos(torch.linspace(0, math.pi, 4))).view(4, 1, 1, 1)
        expected = right * (1 - alpha) + 0.9 * alpha
        torch.testing.assert_close(session.master_frames[4:8], expected)
        torch.testing.assert_close(session.master_frames[:4], torch.full((4, 16, 16, 3), 0.9))

    def test_master_fallback_uses_right_chunk_head(self):
        session = self.make_session()
        right = session.master_frames[8:12].clone()
        self.apply_patch(session)
        self.assert_blend(session, right)
        torch.testing.assert_close(session.master_frames[8:12], right)

    def test_short_right_master_region_skips_blending(self):
        session = self.make_session(master_length=10)
        self.apply_patch(session)
        torch.testing.assert_close(session.master_frames[:8], torch.full((8, 16, 16, 3), 0.9))

    def test_right_patch_takes_precedence(self):
        session = self.make_session()
        right = torch.full((4, 16, 16, 3), 0.7)
        session.has_chunk_patch.return_value = True
        session.load_chunk_patch = Mock(return_value={"frames": right})
        self.apply_patch(session)
        self.assert_blend(session, right)

    def test_unedited_right_chunk_does_not_blend(self):
        session = self.make_session()
        session.meta["chunks"][1]["status"] = "unprocessed"
        self.apply_patch(session)
        torch.testing.assert_close(session.master_frames[:8], torch.full((8, 16, 16, 3), 0.9))


if __name__ == "__main__":
    unittest.main()
