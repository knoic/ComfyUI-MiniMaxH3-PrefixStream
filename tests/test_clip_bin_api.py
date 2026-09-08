"""Unit and integration tests for Clip Bin API and Web endpoints."""

import os
import sys
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from engine.clip_bin_manager import save_clip_asset, get_project_dir
from engine.clip_bin_api import get_project_clips_api, update_clip_rating_api, register_clip_bin_routes


class TestClipBinAPI(unittest.TestCase):
    def setUp(self):
        self.project_name = "Test_API_Project"
        self.p_dir = get_project_dir(self.project_name)
        if os.path.exists(self.p_dir):
            shutil.rmtree(self.p_dir, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.p_dir):
            shutil.rmtree(self.p_dir, ignore_errors=True)

    def test_get_project_clips_api_empty(self):
        data = get_project_clips_api(self.project_name)
        self.assertEqual(data["project_name"], self.project_name)
        self.assertEqual(data["total_clips"], 0)
        self.assertEqual(len(data["clips"]), 0)

    def test_get_project_clips_api_with_assets_and_ratings(self):
        video = torch.randn(1, 16, 32, 88, 160)
        audio = torch.randn(1, 8, 32, 64)
        images = torch.rand(4, 720, 1280, 3)

        # 1. Save clip 1
        meta1, clip_dir1, _ = save_clip_asset(
            video_tensor=video,
            audio_tensor=audio,
            images=images,
            project_name=self.project_name,
            shot_tag="Shot 1_雨夜登场",
            prompt="cyberpunk rain",
            rating=4
        )

        # 2. Save clip 2
        meta2, clip_dir2, _ = save_clip_asset(
            video_tensor=video,
            audio_tensor=audio,
            images=images,
            project_name=self.project_name,
            shot_tag="Shot 2B_拔刀",
            prompt="samurai sword neon",
            rating=5,
            parent_clip_id=meta1.clip_id
        )

        # Test API list
        data = get_project_clips_api(self.project_name)
        self.assertEqual(data["total_clips"], 2)
        clips = data["clips"]
        self.assertEqual(clips[0]["clip_id"], meta2.clip_id)
        self.assertEqual(clips[0]["rating"], 5)
        self.assertEqual(clips[0]["shot_tag"], "Shot 2B_拔刀")
        self.assertEqual(clips[0]["parent_clip_id"], meta1.clip_id)
        self.assertTrue(clips[0]["thumbnail_url"].startswith("/view?filename="))

        # Test rating update API
        success = update_clip_rating_api(self.project_name, meta1.clip_id, 5)
        self.assertTrue(success)

        # Re-fetch and verify rating updated
        data_after = get_project_clips_api(self.project_name)
        clip1_after = next(c for c in data_after["clips"] if c["clip_id"] == meta1.clip_id)
        self.assertEqual(clip1_after["rating"], 5)

    def test_register_routes_safe_without_server(self):
        # Should execute cleanly without raising exception even when server is absent
        register_clip_bin_routes()


if __name__ == "__main__":
    unittest.main()
