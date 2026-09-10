"""Integration unit tests for ComfyUI custom nodes and pipeline."""

import sys
import os
import shutil
import tempfile
import unittest
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

import __init__ as pkg_init
import nodes


class TestNodesAndPipeline(unittest.TestCase):
    def test_node_mappings_consistency(self):
        """Verify all defined nodes in nodes.py are exported in __init__.py."""
        self.assertEqual(set(pkg_init.NODE_CLASS_MAPPINGS.keys()), set(nodes.NODE_CLASS_MAPPINGS.keys()))
        self.assertEqual(set(pkg_init.NODE_DISPLAY_NAME_MAPPINGS.keys()), set(nodes.NODE_DISPLAY_NAME_MAPPINGS.keys()))
        self.assertIn("MiniMaxSaveLatent", pkg_init.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxLoadLatent", pkg_init.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxTrimPrefix", pkg_init.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxLongVideoStitcher", pkg_init.NODE_CLASS_MAPPINGS)

    def test_config_node(self):
        cfg_node = nodes.MiniMaxPrefixCacheConfigNode()
        mode_choices = cfg_node.INPUT_TYPES()["required"]["cache_mode"][0]
        frame_choices = cfg_node.INPUT_TYPES()["required"]["continuation_frames"][0]
        self.assertEqual(mode_choices, [
            "Native Masked AV (Recommended)",
            "Safe Native (Fallback)",
        ])
        self.assertEqual(frame_choices, ["39", "90", "141", "192"])
        (cfg,) = cfg_node.create_config(
            cache_mode="Native Masked AV (Recommended)",
            continuation_frames="39",
        )
        self.assertEqual(cfg.rolling_frames, 39)
        self.assertEqual(cfg.rolling_latent_frames, 12)
        self.assertTrue(cfg.is_native_masked_av_mode())
        self.assertFalse(cfg.is_cache_enabled())

    def test_applier_exposes_only_continuation_inputs(self):
        optional = nodes.MiniMaxPrefixCacheApplierNode.INPUT_TYPES()["optional"]
        self.assertEqual(set(optional), {"cache_config", "context_latent", "target_latent"})

    def test_av_latent_pack_unpack(self):
        v = torch.randn(1, 24, 10, 16, 16)
        a = torch.randn(1, 32, 2, 16)

        latent_dict = nodes.pack_av_latent(v, a)
        unpacked_v, unpacked_a = nodes._unpack_latent(latent_dict)

        self.assertIsNotNone(unpacked_v)
        self.assertEqual(unpacked_v.shape, v.shape)
        self.assertIsNotNone(unpacked_a)
        self.assertEqual(unpacked_a.shape, a.shape)

    def test_save_and_load_latent_node(self):
        temp_dir = tempfile.mkdtemp(prefix="minimax_test_")
        try:
            import types
            dummy_folder_paths = types.ModuleType("folder_paths")
            dummy_folder_paths.get_output_directory = lambda: temp_dir
            sys.modules["folder_paths"] = dummy_folder_paths

            v = torch.randn(1, 24, 7, 16, 16)
            a = torch.randn(1, 32, 2, 11)
            original_latent = nodes.pack_av_latent(v, a)

            save_node = nodes.MiniMaxSaveLatentNode()
            saved_path, info = save_node.save(
                latent=original_latent,
                filename_prefix="test_clip/seg",
                clip_index=1
            )
            self.assertTrue(os.path.exists(saved_path))
            self.assertIn("frames", info)

            load_node = nodes.MiniMaxLoadLatentNode()
            loaded_latent, loaded_path, load_info = load_node.load(
                latent_path=saved_path,
                clip_index=1
            )
            self.assertEqual(loaded_path, saved_path)
            lv, la = nodes._unpack_latent(loaded_latent)
            self.assertEqual(lv.shape, v.shape)
            if la is not None:
                self.assertEqual(la.shape, a.shape)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_trim_node_pixel_and_waveform(self):
        trim_node = nodes.MiniMaxTrimPrefixLatentNode()
        images = torch.ones((48, 32, 32, 3), dtype=torch.float32)
        audio = {"waveform": torch.ones((1, 2, 64000)), "sample_rate": 32000}

        out_img, out_aud, _ = trim_node.trim(
            trim_frames=24,
            images=images,
            audio=audio,
            fps=24.0
        )
        self.assertEqual(out_img.shape[0], 24)
        self.assertEqual(out_aud["waveform"].shape[-1], 32000)

    def test_trim_node_initial_clip_guard(self):
        """When cache_config is provided but session is None and no noise_mask is present, preserve initial clip."""
        trim_node = nodes.MiniMaxTrimPrefixLatentNode()
        cfg_node = nodes.MiniMaxPrefixCacheConfigNode()
        (cfg,) = cfg_node.create_config(continuation_frames="39")

        images = torch.ones((50, 32, 32, 3), dtype=torch.float32)
        v = torch.randn(1, 24, 15, 16, 16)
        latent_initial = nodes.pack_av_latent(v, None)

        out_img, _, _ = trim_node.trim(
            trim_frames=0,  # auto mode
            images=images,
            latent=latent_initial,
            cache_config=cfg,
            session=None,
        )
        # Should preserve all 50 frames because it is initial clip
        self.assertEqual(out_img.shape[0], 50)

    def test_long_video_stitcher_node(self):
        stitcher = nodes.MiniMaxLongVideoStitcherNode()
        prev_img = torch.ones((30, 64, 64, 3), dtype=torch.float32)
        curr_img = torch.ones((25, 64, 64, 3), dtype=torch.float32)
        prev_aud = {"waveform": torch.zeros((1, 2, 32000)), "sample_rate": 32000}
        curr_aud = {"waveform": torch.zeros((1, 2, 32000)), "sample_rate": 32000}

        out_img, out_aud, total_f = stitcher.stitch(
            trim_frames=10,
            crossfade_frames=2,
            prev_images=prev_img,
            curr_images=curr_img,
            prev_audio=prev_aud,
            curr_audio=curr_aud,
            fps=24.0,
        )
        self.assertEqual(out_img.shape[0], 45)
        self.assertEqual(total_f, 45)
        self.assertIsNotNone(out_aud)
        self.assertIn("waveform", out_aud)

    def test_clip_bin_tree_picker_node(self):
        # 1. Verify existence and inheritance
        self.assertIn("MiniMaxClipBinTreePicker", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxClipBinTreePicker", nodes.NODE_DISPLAY_NAME_MAPPINGS)
        tree_node = nodes.MiniMaxClipBinTreePickerNode()
        picker_node = nodes.MiniMaxClipBinPickerNode()

        # 2. Check input types and widget order
        picker_types = picker_node.INPUT_TYPES()
        tree_types = tree_node.INPUT_TYPES()

        # Both must keep required fields identically ordered
        self.assertEqual(list(picker_types["required"].keys()), ["project_name", "mode", "filter_rating", "clip_selection"])
        self.assertEqual(list(tree_types["required"].keys()), ["project_name", "mode", "filter_rating", "clip_selection"])

        # view_mode must be the last optional item
        self.assertEqual(list(picker_types["optional"].keys())[-1], "view_mode")
        self.assertEqual(list(tree_types["optional"].keys())[-1], "view_mode")

        # Check default view_mode settings
        self.assertEqual(picker_types["optional"]["view_mode"][1]["default"], "Deck (卡片流)")
        self.assertEqual(tree_types["optional"]["view_mode"][1]["default"], "Tree (关系树)")

        # 3. Test backward-compatibility: call without view_mode (as old workflow would)
        res1 = picker_node.pick_clip(project_name="Test_Empty_Proj_1", mode="Auto (首段全新 / 后续自动接力)")
        self.assertIn("result", res1)
        self.assertEqual(res1["result"][4], "[INITIAL_GENERATION]")

        # 4. Call tree picker with view_mode explicitly passed
        res2 = tree_node.pick_clip(project_name="Test_Empty_Proj_2", mode="Auto (首段全新 / 后续自动接力)", view_mode="Tree (关系树)")
        self.assertIn("result", res2)
        self.assertEqual(res2["result"][4], "[INITIAL_GENERATION]")

    def test_workflow_graph_integrity(self):
        import json
        wf_file = os.path.join(_repo_dir, "examples", "MiniMaxH3_PrefixStream_v1.0.json")
        self.assertTrue(os.path.exists(wf_file), f"Workflow file not found at {wf_file}")

        with open(wf_file, "r", encoding="utf-8") as f:
            wf = json.load(f)
        nodes_dict = {n["id"]: n for n in wf.get("nodes", [])}
        links_dict = {l[0]: l for l in wf.get("links", []) if l}
        for nid, n in nodes_dict.items():
            for slot, inp in enumerate(n.get("inputs", [])):
                lid = inp.get("link")
                if lid is not None:
                    self.assertIn(lid, links_dict, f"Link {lid} on Node {nid} slot {slot} missing from global links!")
                    l = links_dict[lid]
                    self.assertEqual((l[3], l[4]), (nid, slot), f"Link {lid} target mismatch!")


if __name__ == "__main__":
    unittest.main()
