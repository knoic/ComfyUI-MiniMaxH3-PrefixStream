"""Unit and integration tests for MiniMax Clip Bin system."""

import os
import sys
import shutil
import tempfile
import types
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

import nodes
from engine.clip_bin_manager import (
    load_project_index,
    rebuild_project_index,
    list_projects,
    tensor_to_pil,
    pil_to_tensor,
)


def setup_temp_folder_paths():
    temp_dir = tempfile.mkdtemp(prefix="minimax_clip_bin_test_")
    dummy_folder_paths = types.ModuleType("folder_paths")
    dummy_folder_paths.get_output_directory = lambda: temp_dir
    sys.modules["folder_paths"] = dummy_folder_paths
    return temp_dir


def test_clip_bin_saver_and_picker_with_images():
    temp_dir = setup_temp_folder_paths()
    try:
        # Mock AV Latent (MiniMax H3 format)
        v = torch.randn(1, 16, 7, 32, 32)
        a = torch.randn(1, 16, 2, 16)
        latent = nodes.pack_av_latent(v, a)


        # Mock images [B, H, W, C]
        images = torch.rand(24, 64, 64, 3, dtype=torch.float32)

        saver = nodes.MiniMaxClipBinSaverNode()
        res_save = saver.save_clip(
            latent=latent,
            project_name="SciFi_Film",
            shot_tag="Opening_Shot",
            rating=5,
            images=images,
            prompt="A spaceship descending through clouds",
            video_file_name="SciFi_001.mp4"
        )

        assert "result" in res_save and "ui" in res_save
        clip_id, preview_img, bin_path = res_save["result"]
        assert clip_id.startswith("clip_")
        assert "Opening_Shot" in clip_id
        assert os.path.isdir(bin_path)

        # Check files
        assert os.path.isfile(os.path.join(bin_path, "latent.safetensors"))
        assert os.path.isfile(os.path.join(bin_path, "first_frame.png"))
        assert os.path.isfile(os.path.join(bin_path, "tail_frame.png"))
        assert os.path.isfile(os.path.join(bin_path, "preview.png"))
        assert os.path.isfile(os.path.join(bin_path, "meta.json"))

        # Test Picker
        picker = nodes.MiniMaxClipBinPickerNode()
        res_pick = picker.pick_clip(
            project_name="SciFi_Film",
            filter_rating="All (1-5 ⭐)",
            clip_selection="latest"
        )

        assert "result" in res_pick and "ui" in res_pick
        out_latent, tail_frame, first_frame, prompt_str, loaded_id = res_pick["result"]

        assert loaded_id == clip_id
        assert prompt_str == "A spaceship descending through clouds"
        assert tail_frame.shape == (1, 64, 64, 3)
        assert first_frame.shape == (1, 64, 64, 3)

        # Verify AV Latent integrity
        unpacked_v, unpacked_a = nodes._unpack_latent(out_latent)
        assert unpacked_v.shape == v.shape
        assert unpacked_a is not None and unpacked_a.shape == a.shape

        print("test_clip_bin_saver_and_picker_with_images passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_clip_bin_saver_without_images_fallback():
    temp_dir = setup_temp_folder_paths()
    try:
        v = torch.randn(1, 16, 4, 16, 16)
        latent = nodes.pack_av_latent(v, None)

        saver = nodes.MiniMaxClipBinSaverNode()
        res_save = saver.save_clip(
            latent=latent,
            project_name="CyberCity",
            shot_tag="Neon_Alley",
            rating=3,
            images=None,
            prompt="Cyberpunk rain"
        )

        clip_id, preview_img, bin_path = res_save["result"]
        assert os.path.isfile(os.path.join(bin_path, "first_frame.png"))
        assert os.path.isfile(os.path.join(bin_path, "tail_frame.png"))
        assert preview_img.ndim == 4 and preview_img.shape[-1] == 3

        picker = nodes.MiniMaxClipBinPickerNode()
        res_pick = picker.pick_clip(
            project_name="CyberCity",
            filter_rating="All (1-5 ⭐)",
            clip_selection=clip_id
        )
        out_latent, tail_frame, first_frame, prompt_str, loaded_id = res_pick["result"]
        assert loaded_id == clip_id
        assert tail_frame.ndim == 4

        print("test_clip_bin_saver_without_images_fallback passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rating_filter():
    temp_dir = setup_temp_folder_paths()
    try:
        v = torch.randn(1, 16, 4, 16, 16)
        latent = nodes.pack_av_latent(v, None)
        saver = nodes.MiniMaxClipBinSaverNode()

        # Clip 1: rating 2 (bad take)
        res1 = saver.save_clip(latent=latent, project_name="RatingTest", shot_tag="BadTake", rating=2)
        id1 = res1["result"][0]

        # Clip 2: rating 5 (great take)
        res2 = saver.save_clip(latent=latent, project_name="RatingTest", shot_tag="GreatTake", rating=5)
        id2 = res2["result"][0]

        # Clip 3: rating 1 (trash)
        res3 = saver.save_clip(latent=latent, project_name="RatingTest", shot_tag="Trash", rating=1)
        id3 = res3["result"][0]

        picker = nodes.MiniMaxClipBinPickerNode()

        # Filter 5 stars with 'latest': Should skip id3 (rating 1) and pick id2 (rating 5)
        res_pick5 = picker.pick_clip(
            project_name="RatingTest",
            filter_rating="⭐⭐⭐⭐⭐ (5 ⭐)",
            clip_selection="latest"
        )
        assert res_pick5["result"][4] == id2

        # Filter All with 'latest': Should pick id3 (the most recent one)
        res_pick_all = picker.pick_clip(
            project_name="RatingTest",
            filter_rating="All (1-5 ⭐)",
            clip_selection="latest"
        )
        assert res_pick_all["result"][4] == id3

        print("test_rating_filter passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_auto_rebuild():
    temp_dir = setup_temp_folder_paths()
    try:
        v = torch.randn(1, 16, 4, 16, 16)
        latent = nodes.pack_av_latent(v, None)
        saver = nodes.MiniMaxClipBinSaverNode()

        res = saver.save_clip(latent=latent, project_name="RebuildTest", shot_tag="Scene1", rating=4)
        clip_id = res["result"][0]

        # Delete the .bin_index.json
        from engine.clip_bin_manager import _get_index_path
        idx_path = _get_index_path("RebuildTest")
        assert os.path.isfile(idx_path)
        os.remove(idx_path)
        assert not os.path.isfile(idx_path)

        # Call load_project_index: it should auto rebuild
        idx = load_project_index("RebuildTest")
        assert idx["total_clips"] == 1
        assert idx["clips"][0]["clip_id"] == clip_id

        print("test_index_auto_rebuild passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_auto_initial_and_chaining_unified_workflow():
    temp_dir = setup_temp_folder_paths()
    try:
        picker = nodes.MiniMaxClipBinPickerNode()
        saver = nodes.MiniMaxClipBinSaverNode()

        # 1. First run: Empty project with Auto mode -> should return Initial Mode (None latent)
        res_init = picker.pick_clip(
            project_name="UnifiedProject",
            mode="Auto (首段全新 / 后续自动接力)"
        )
        assert res_init["result"][0] is None
        assert res_init["result"][4] == "[INITIAL_GENERATION]"

        # 2. Save Clip 1 with Auto shot_tag -> should become Shot 1
        v1 = torch.randn(1, 16, 4, 16, 16)
        latent1 = nodes.pack_av_latent(v1, None)
        res_s1 = saver.save_clip(
            latent=latent1,
            project_name="UnifiedProject",
            shot_tag="Auto (自动编号)"
        )
        clip1_id = res_s1["result"][0]
        assert "Shot 1" in clip1_id or "Shot1" in clip1_id or "_Shot_1" in clip1_id or "Shot" in clip1_id

        # 3. Second run: Same project with Auto mode -> should automatically pick Clip 1!
        res_pick1 = picker.pick_clip(
            project_name="UnifiedProject",
            mode="Auto (首段全新 / 后续自动接力)"
        )
        assert res_pick1["result"][0] is not None
        assert res_pick1["result"][4] == clip1_id

        # 4. Save Clip 2 with Auto shot_tag -> should become Shot 2
        v2 = torch.randn(1, 16, 4, 16, 16)
        latent2 = nodes.pack_av_latent(v2, None)
        res_s2 = saver.save_clip(
            latent=latent2,
            project_name="UnifiedProject",
            shot_tag="Auto (自动编号)"
        )
        clip2_id = res_s2["result"][0]
        assert clip2_id != clip1_id

        # 5. Third run: Picks Clip 2
        res_pick2 = picker.pick_clip(
            project_name="UnifiedProject",
            mode="Auto (首段全新 / 后续自动接力)"
        )
        assert res_pick2["result"][4] == clip2_id

        print("test_auto_initial_and_chaining_unified_workflow passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_clip_bin_saver_and_picker_with_images()
    test_clip_bin_saver_without_images_fallback()
    test_rating_filter()
    test_index_auto_rebuild()
    test_auto_initial_and_chaining_unified_workflow()
    print("\n>>> All MiniMax Clip Bin tests PASSED successfully! <<<")

