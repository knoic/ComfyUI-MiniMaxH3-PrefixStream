"""Unit tests for MiniMax H3 Video Chunk Slicer & Timeline Patch Reassembler."""

import sys
import os
import shutil
import tempfile
import unittest
import subprocess
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

import __init__ as pkg_init
import nodes
from engine.timeline_session_manager import (
    TimelineSession,
    get_or_create_timeline_session,
    slice_video_and_audio,
    render_timeline_indicator_image,
)


class TestTimelineSlicer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_timeline_")
        self.orig_base = os.environ.get("MINIMAX_CLIP_BIN_DIR")
        os.environ["MINIMAX_CLIP_BIN_DIR"] = self.test_dir

    def tearDown(self):
        if self.orig_base is not None:
            os.environ["MINIMAX_CLIP_BIN_DIR"] = self.orig_base
        else:
            os.environ.pop("MINIMAX_CLIP_BIN_DIR", None)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_node_mappings(self):
        """Verify new nodes are properly exported in both nodes.py and __init__.py."""
        self.assertIn("MiniMaxVideoChunkSlicer", pkg_init.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxVideoPatchReassembler", pkg_init.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxVideoChunkSlicer", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxVideoPatchReassembler", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("MiniMaxVideoChunkSlicer", pkg_init.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertIn("MiniMaxVideoPatchReassembler", pkg_init.NODE_DISPLAY_NAME_MAPPINGS)

    def test_zero_drift_slicing(self):
        """Verify strict integer frame slicing and sample-accurate audio locking."""
        total_frames = 300
        fps = 24.0
        sr = 44100
        total_audio_samples = int(round(total_frames / fps * sr))

        images = torch.ones((total_frames, 64, 64, 3), dtype=torch.float32)
        audio = {
            "waveform": torch.randn((2, total_audio_samples), dtype=torch.float32),
            "sample_rate": sr,
        }

        # Chunk 0: 0 ~ 124
        c0_img, c0_aud, ctx0, info0 = slice_video_and_audio(
            images=images, audio=audio, fps=fps, project_name="UnitTest_ZeroDrift",
            chunk_length=124, chunk_index=0
        )
        self.assertEqual(c0_img.shape[0], 124)
        self.assertEqual(ctx0["start_frame"], 0)
        self.assertEqual(ctx0["end_frame"], 124)
        expected_s0 = int(round(124 / fps * sr))
        self.assertEqual(c0_aud["waveform"].shape[-1], expected_s0)

        # Chunk 1: 124 ~ 248
        c1_img, c1_aud, ctx1, info1 = slice_video_and_audio(
            images=images, audio=audio, fps=fps, project_name="UnitTest_ZeroDrift",
            chunk_length=124, chunk_index=1
        )
        self.assertEqual(c1_img.shape[0], 124)
        self.assertEqual(ctx1["start_frame"], 124)
        self.assertEqual(ctx1["end_frame"], 248)

        # Chunk 2: 248 ~ 300 (Tail chunk, 52 frames)
        c2_img, c2_aud, ctx2, info2 = slice_video_and_audio(
            images=images, audio=audio, fps=fps, project_name="UnitTest_ZeroDrift",
            chunk_length=124, chunk_index=2
        )
        self.assertEqual(c2_img.shape[0], 52)
        self.assertEqual(ctx2["start_frame"], 248)
        self.assertEqual(ctx2["end_frame"], 300)

        # Total sliced frames sum to exactly total_frames with zero drift
        self.assertEqual(c0_img.shape[0] + c1_img.shape[0] + c2_img.shape[0], total_frames)

    def test_in_place_patch_and_gap_detection(self):
        """Verify in-place patching, session assembly, and gap detection."""
        total_frames = 300
        fps = 24.0
        images = torch.zeros((total_frames, 32, 32, 3), dtype=torch.float32)  # All zeros

        session = get_or_create_timeline_session("UnitTest_PatchGap")
        session.initialize_base(images=images, audio=None, fps=fps, chunk_length=100)

        self.assertEqual(len(session.meta["chunks"]), 3)  # 0~100, 100~200, 200~300
        self.assertEqual(session.meta["coverage_ratio"], 0.0)

        # 1. Patch Chunk 0 with all ones
        patch0 = torch.ones((100, 32, 32, 3), dtype=torch.float32)
        session.patch_chunk(chunk_index=0, start_frame=0, end_frame=100, edited_images=patch0, seam_blend_frames=0)
        self.assertAlmostEqual(session.meta["coverage_ratio"], 100 / 300.0, places=3)
        self.assertFalse(session.meta["is_fully_assembled"])
        # Master frames [0:100] should be 1.0, and [100:300] should be 0.0
        self.assertTrue(torch.allclose(session.master_frames[:100], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(session.master_frames[100:], torch.tensor(0.0)))

        # 2. Skip Chunk 1 and patch Chunk 2 (Intentionally leave gap)
        patch2 = torch.full((100, 32, 32, 3), 2.0, dtype=torch.float32)
        session.patch_chunk(chunk_index=2, start_frame=200, end_frame=300, edited_images=patch2, seam_blend_frames=0)
        self.assertFalse(session.meta["is_fully_assembled"])

        # Check gap list: Chunk 1 (100~200) MUST be identified as uncompleted gap
        gaps = session.meta["gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["chunk_index"], 1)
        self.assertEqual(gaps[0]["start_frame"], 100)
        self.assertEqual(gaps[0]["end_frame"], 200)

        # 3. Patch the missing Chunk 1 to complete the long video
        patch1 = torch.full((100, 32, 32, 3), 1.5, dtype=torch.float32)
        session.patch_chunk(chunk_index=1, start_frame=100, end_frame=200, edited_images=patch1, seam_blend_frames=0)

        # Verify 100% assembly complete
        self.assertEqual(session.meta["coverage_ratio"], 1.0)
        self.assertTrue(session.meta["is_fully_assembled"])
        self.assertEqual(len(session.meta["gaps"]), 0)

    def test_node_execution_pipeline(self):
        """End-to-end execution test of MiniMaxVideoChunkSlicerNode and MiniMaxVideoPatchReassemblerNode."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        reassembler = nodes.MiniMaxVideoPatchReassemblerNode()

        full_video = torch.randn((150, 48, 48, 3), dtype=torch.float32)
        p_name = "UnitTest_Pipeline"

        # Slice Chunk 0 via connected images
        chunk_imgs, chunk_aud, ctx, v_info, preview, fps_val, frame_cnt, info, prev_last_f, prev_ref_fs = slicer.slice_chunk(
            video_file="none",
            project_name=p_name,
            chunk_length=124,
            chunk_index=0,
            images=full_video,
        )
        self.assertEqual(chunk_imgs.shape[0], 124)
        self.assertIsNotNone(preview)
        self.assertIn("UnitTest_Pipeline", info)
        self.assertEqual(fps_val, 24.0)
        self.assertEqual(frame_cnt, 124)
        self.assertIn("source_fps", v_info)
        self.assertIn("loaded_frame_count", v_info)
        self.assertEqual(v_info["source_frame_count"], 150)
        self.assertEqual(v_info["loaded_frame_count"], 124)

        # Reassemble
        out_imgs, out_aud, is_done, summary = reassembler.reassemble_patch(
            edited_images=chunk_imgs,
            slice_context=ctx,
            seam_blend_frames=2,
            output_mode="Full Assembled Video (完整拼装长视频)",
        )

        # Master video shape matches original 150 frames
        self.assertEqual(out_imgs.shape[0], 150)
        self.assertFalse(is_done)  # Still has 26 frames in chunk 1 unedited
        self.assertIn("尚未完成", summary)

    def test_direct_video_file_lazy_loading(self):
        """Verify direct video file ingest via ffmpeg without full-RAM pre-decoding."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            self.skipTest("ffmpeg not available")

        # Create a synthetic 60-frame mp4 video
        video_path = os.path.join(self.test_dir, "synth_test.mp4")
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2.5:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2.5",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.isfile(video_path):
            self.skipTest("ffmpeg synthetic video generation failed")

        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        # Extract chunk 0 (frames 0~30) with target scaling to 640x360
        c_imgs, c_aud, ctx, v_info, preview, fps_val, frame_cnt, info, *extra = slicer.slice_chunk(
            video_file=video_path,
            project_name="UnitTest_FileIngest",
            chunk_length=30,
            chunk_index=0,
            target_width=640,
            target_height=360,
            force_fps=24.0,
            images=None,
        )

        self.assertEqual(c_imgs.shape[0], 30)
        self.assertEqual(c_imgs.shape[1], 360)
        self.assertEqual(c_imgs.shape[2], 640)
        self.assertEqual(ctx["start_frame"], 0)
        self.assertEqual(ctx["end_frame"], 30)
        self.assertIsNotNone(c_aud)
        self.assertEqual(c_aud["sample_rate"], 44100)
        self.assertIn("source_fps", v_info)
        self.assertEqual(v_info["loaded_frame_count"], 30)
        self.assertEqual(v_info["loaded_width"], 640)
        self.assertEqual(v_info["loaded_height"], 360)

        # Test video probing
        from engine.timeline_session_manager import probe_video_info
        probe = probe_video_info(video_path)
        self.assertGreaterEqual(probe["duration"], 2.0)
        self.assertEqual(probe["width"], 320)
        self.assertEqual(probe["height"], 240)
        self.assertTrue(probe["has_audio"])

    def test_export_master_video_and_auto_advance(self):
        """Verify export_master_to_video_file exports clean mp4 and auto_advance stores in ctx."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            self.skipTest("ffmpeg not available")

        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        reassembler = nodes.MiniMaxVideoPatchReassemblerNode()

        full_video = torch.randn((60, 48, 48, 3), dtype=torch.float32)
        p_name = "UnitTest_Export"

        chunk_imgs, chunk_aud, ctx, v_info, preview, fps_val, frame_cnt, info, *extra = slicer.slice_chunk(
            video_file="none",
            project_name=p_name,
            chunk_length=30,
            chunk_index=0,
            auto_advance="Next Chunk (顺序下一段)",
            images=full_video,
        )
        self.assertEqual(ctx["auto_advance"], "Next Chunk (顺序下一段)")

        # Reassemble
        out_imgs, out_aud, is_done, summary = reassembler.reassemble_patch(
            edited_images=chunk_imgs,
            slice_context=ctx,
            seam_blend_frames=2,
            output_mode="Full Assembled Video (完整拼装长视频)",
        )

        from engine.timeline_session_manager import export_master_to_video_file
        res = export_master_to_video_file(
            project_name=p_name,
            output_dir=self.test_dir,
            filename="exported_test.mp4",
        )
        self.assertTrue(res["success"])
        self.assertTrue(os.path.isfile(res["file_path"]))
        self.assertGreaterEqual(res["file_size_mb"], 0.0)
        self.assertEqual(res["total_frames"], 60)

    def test_legacy_shifted_widget_values_resilience(self):
        """Verify slice_chunk handles legacy shifted inputs where auto_advance was 24 and fps was invalid."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        full_video = torch.randn((30, 48, 48, 3), dtype=torch.float32)

        # VALIDATE_INPUTS must return True
        self.assertTrue(nodes.MiniMaxVideoChunkSlicerNode.VALIDATE_INPUTS(auto_advance=24, fps="invalid_fps"))

        # slice_chunk must not crash and fallback to safe defaults
        chunk_imgs, chunk_aud, ctx, v_info, preview, fps_val, frame_cnt, info, *extra = slicer.slice_chunk(
            video_file="none",
            project_name="UnitTest_Resilience",
            chunk_length=30,
            chunk_index=0,
            auto_advance=24,  # Legacy shifted value
            fps="unconvertible",  # Legacy shifted value
            images=full_video,
        )
        self.assertEqual(ctx["auto_advance"], "None (手动控制)")
        self.assertEqual(fps_val, 24.0)
        self.assertEqual(chunk_imgs.shape[0], 30)

    def test_audio_vae_encode_compatibility_and_fallback_silence(self):
        """Regression test for Node 526 NKDAVLatent / VAEEncodeAudio IndexError: tuple index out of range."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        dummy_img = torch.ones((124, 64, 64, 3), dtype=torch.float32)

        # Case 1: Slicing with NO input audio -> must synthesize valid 3D silent audio
        out = slicer.slice_chunk(
            video_file="",
            project_name="UnitTest_AudioCompat",
            chunk_length=124,
            chunk_index=0,
            images=dummy_img,
            audio=None,
        )
        chunk_aud = out[1]
        self.assertIsNotNone(chunk_aud)
        self.assertIn("waveform", chunk_aud)
        self.assertIn("sample_rate", chunk_aud)
        self.assertEqual(chunk_aud["waveform"].ndim, 3)
        self.assertEqual(chunk_aud["waveform"].shape[0], 1)
        self.assertEqual(chunk_aud["waveform"].shape[1], 2)

        # Verify movedim(1, -1) produces [1, samples, channels] where shape[2] is valid (e.g. 2)
        moved = chunk_aud["waveform"].movedim(1, -1)
        self.assertEqual(len(moved.shape), 3)
        self.assertEqual(moved.shape[2], 2)

        # Case 2: Slicing with 2D audio input -> auto converted to 3D
        audio_2d = {"waveform": torch.zeros((2, 44100)), "sample_rate": 44100}
        out_2d = slicer.slice_chunk(
            video_file="",
            project_name="UnitTest_AudioCompat",
            chunk_length=124,
            chunk_index=0,
            images=dummy_img,
            audio=audio_2d,
        )
        self.assertEqual(out_2d[1]["waveform"].ndim, 3)
        self.assertEqual(out_2d[1]["waveform"].shape[0], 1)

        # Case 3: Reassembler output audio must also be valid 3D
        reassembler = nodes.MiniMaxVideoPatchReassemblerNode()
        ctx = out[2]
        res = reassembler.reassemble_patch(
            edited_images=dummy_img,
            slice_context=ctx,
            edited_audio=chunk_aud,
        )
        out_aud = res[1]
        self.assertIsNotNone(out_aud)
        self.assertEqual(out_aud["waveform"].ndim, 3)
        self.assertEqual(out_aud["waveform"].shape[0], 1)
        self.assertEqual(len(out_aud["waveform"].movedim(1, -1).shape), 3)
        self.assertEqual(out_aud["waveform"].movedim(1, -1).shape[2], 2)

    def test_patch_resolution_mismatch_auto_resize(self):
        """Verify that patch_chunk and Reassembler gracefully auto-resize when edited patch has different resolution."""
        # 1. Master video initialized with 3840x2160 resolution (simulated with smaller proportionally: 64x48)
        total_frames = 100
        master_img = torch.zeros((total_frames, 48, 64, 3), dtype=torch.float32)
        session = get_or_create_timeline_session("UnitTest_ResMismatch")
        session.initialize_base(images=master_img, audio=None, fps=24.0, chunk_length=50)

        # 2. Downstream model outputted a different resolution: e.g. 32x40 (simulating 2688x1512 vs 3840x2160)
        edited_patch = torch.ones((50, 32, 40, 3), dtype=torch.float32)

        # 3. Patch chunk with seam blending enabled (seam_blend_frames=4)
        # Before fix: crashed with RuntimeError: The size of tensor a (64) must match the size of tensor b (40)
        updated_chunk = session.patch_chunk(
            chunk_index=0,
            start_frame=0,
            end_frame=50,
            edited_images=edited_patch,
            seam_blend_frames=4,
        )
        self.assertIsNotNone(updated_chunk)
        self.assertEqual(updated_chunk["status"], "completed")
        self.assertEqual(session.master_frames.shape, (100, 48, 64, 3))

        # 4. Also test via MiniMaxVideoPatchReassemblerNode
        reassembler = nodes.MiniMaxVideoPatchReassemblerNode()
        ctx = {
            "project_name": "UnitTest_ResMismatch",
            "chunk_index": 1,
            "start_frame": 50,
            "end_frame": 100,
            "frame_count": 50,
            "chunk_length": 50,
            "total_frames": 100,
            "fps": 24.0,
            "width": 64,
            "height": 48,
        }
        # Edited chunk with yet another resolution (e.g. 24x36)
        patch_2 = torch.full((50, 24, 36, 3), 0.8, dtype=torch.float32)
        out_imgs, out_aud, is_done, summary = reassembler.reassemble_patch(
            edited_images=patch_2,
            slice_context=ctx,
            seam_blend_frames=4,
            output_mode="Full Assembled Video (完整拼装长视频)",
        )
        self.assertTrue(is_done)
        self.assertEqual(out_imgs.shape, (100, 48, 64, 3))

    def test_slicer_node_outputs_count_and_types(self):
        """Verify MiniMaxVideoChunkSlicerNode exports 10 returns with prev_last_frame and prev_ref_frames."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        self.assertEqual(len(slicer.RETURN_TYPES), 10)
        self.assertEqual(len(slicer.RETURN_NAMES), 10)
        self.assertEqual(slicer.RETURN_NAMES[8], "prev_last_frame")
        self.assertEqual(slicer.RETURN_NAMES[9], "prev_ref_frames")
        self.assertEqual(slicer.RETURN_TYPES[8], "IMAGE")
        self.assertEqual(slicer.RETURN_TYPES[9], "IMAGE")

        # Check input types
        inputs = slicer.INPUT_TYPES()
        self.assertIn("prev_ref_frames_count", inputs["required"])
        self.assertIn("first_chunk_ref_mode", inputs["required"])
        self.assertIn("optional_first_frame_ref", inputs["optional"])

    def test_previous_reference_frames_first_chunk_fallback(self):
        """Verify fallback behavior when slicing chunk 0 (no previous chunk)."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        full_video = torch.arange(60).view(60, 1, 1, 1).repeat(1, 32, 32, 3).float() / 60.0

        # Mode A: Current Chunk First Frame (default)
        out = slicer.slice_chunk(
            video_file="none",
            project_name="UnitTest_FirstChunkFallback",
            chunk_length=30,
            chunk_index=0,
            prev_ref_frames_count=10,
            first_chunk_ref_mode="Current Chunk First Frame (当前片段首帧)",
            images=full_video,
        )
        c_imgs, c_aud, ctx, v_info, preview, fps_val, frame_cnt, info, prev_last_f, prev_ref_fs = out
        self.assertEqual(prev_last_f.shape, (1, 32, 32, 3))
        self.assertEqual(prev_ref_fs.shape, (10, 32, 32, 3))
        # Matches first frame of chunk 0
        self.assertTrue(torch.allclose(prev_last_f, c_imgs[0:1]))
        self.assertFalse(ctx["prev_ref_info"]["has_prev_chunk"])
        self.assertFalse(ctx["prev_ref_info"]["is_edited"])
        self.assertIn("首段第0帧", info)

        # Mode B: Black / Zero Frame
        out_b = slicer.slice_chunk(
            video_file="none",
            project_name="UnitTest_FirstChunkFallback",
            chunk_length=30,
            chunk_index=0,
            prev_ref_frames_count=10,
            first_chunk_ref_mode="Black / Zero Frame (全黑空帧)",
            images=full_video,
        )
        _, _, _, _, _, _, _, _, prev_last_b, prev_ref_b = out_b
        self.assertTrue(torch.allclose(prev_last_b, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(prev_ref_b, torch.tensor(0.0)))

        # Mode C: optional_first_frame_ref provided
        custom_ref = torch.full((1, 32, 32, 3), 0.77, dtype=torch.float32)
        out_c = slicer.slice_chunk(
            video_file="none",
            project_name="UnitTest_FirstChunkFallback",
            chunk_length=30,
            chunk_index=0,
            prev_ref_frames_count=8,
            images=full_video,
            optional_first_frame_ref=custom_ref,
        )
        _, _, _, _, _, _, _, _, prev_last_c, prev_ref_c = out_c
        self.assertTrue(torch.allclose(prev_last_c, torch.tensor(0.77)))
        self.assertEqual(prev_ref_c.shape, (8, 32, 32, 3))
        self.assertTrue(torch.allclose(prev_ref_c, torch.tensor(0.77)))

    def test_previous_reference_frames_edited_propagation(self):
        """Verify that after chunk 0 is edited and reassembled, chunk 1 gets edited frames as reference."""
        slicer = nodes.MiniMaxVideoChunkSlicerNode()
        reassembler = nodes.MiniMaxVideoPatchReassemblerNode()
        p_name = "UnitTest_PrevRefEdited"

        orig_video = torch.zeros((100, 32, 32, 3), dtype=torch.float32)  # Raw video is all zeros

        # 1. Slice Chunk 0
        c0_imgs, c0_aud, ctx0, _, _, _, _, _, _, _ = slicer.slice_chunk(
            video_file="none",
            project_name=p_name,
            chunk_length=50,
            chunk_index=0,
            images=orig_video,
        )

        # 2. Simulate model editing Chunk 0: distinct value 0.95
        edited_c0 = torch.full((50, 32, 32, 3), 0.95, dtype=torch.float32)
        reassembler.reassemble_patch(
            edited_images=edited_c0,
            slice_context=ctx0,
            seam_blend_frames=0,
            save_to_session=True,
        )

        # 3. Slice Chunk 1 (frames 50~100)
        c1_imgs, c1_aud, ctx1, _, _, _, _, info1, prev_last_f1, prev_ref_fs1 = slicer.slice_chunk(
            video_file="none",
            project_name=p_name,
            chunk_length=50,
            chunk_index=1,
            prev_ref_frames_count=16,
            images=orig_video,
        )

        # Reference frames MUST be the edited frames (0.95), NOT the original unedited zeros!
        self.assertEqual(prev_last_f1.shape, (1, 32, 32, 3))
        self.assertTrue(torch.allclose(prev_last_f1, torch.tensor(0.95)))
        self.assertEqual(prev_ref_fs1.shape, (16, 32, 32, 3))
        self.assertTrue(torch.allclose(prev_ref_fs1, torch.tensor(0.95)))

        # Metadata checks
        self.assertTrue(ctx1["prev_ref_info"]["has_prev_chunk"])
        self.assertTrue(ctx1["prev_ref_info"]["is_edited"])
        self.assertEqual(ctx1["prev_ref_info"]["prev_frame_index"], 49)
        self.assertIn("已编辑成果 ✅", info1)

    def test_chunk_length_change_and_reload_consistency(self):
        """Verify that modifying chunk_length dynamically recalculates all chunk boundaries without drift."""
        p_name = "UnitTest_ChunkLengthChange"
        total_frames = 1608
        fps = 24.0
        images = torch.zeros((total_frames, 32, 32, 3), dtype=torch.float32)

        session = get_or_create_timeline_session(p_name)
        session.initialize_base(images=images, audio=None, fps=fps, chunk_length=124)

        self.assertEqual(session.meta["chunk_length"], 124)
        self.assertEqual(len(session.meta["chunks"]), 13)
        self.assertEqual(session.meta["chunks"][0]["start_frame"], 0)
        self.assertEqual(session.meta["chunks"][0]["end_frame"], 124)
        self.assertEqual(session.meta["chunks"][3]["start_frame"], 372)
        self.assertEqual(session.meta["chunks"][3]["end_frame"], 496)

        # Re-initialize / change chunk_length to 90 (as reported in user bug)
        session.initialize_base(images=images, audio=None, fps=fps, chunk_length=90)
        self.assertEqual(session.meta["chunk_length"], 90)
        self.assertEqual(len(session.meta["chunks"]), 18)
        self.assertEqual(session.meta["chunks"][0]["start_frame"], 0)
        self.assertEqual(session.meta["chunks"][0]["end_frame"], 90)
        self.assertEqual(session.meta["chunks"][0]["frame_count"], 90)
        self.assertEqual(session.meta["chunks"][1]["start_frame"], 90)
        self.assertEqual(session.meta["chunks"][1]["end_frame"], 180)
        self.assertEqual(session.meta["chunks"][3]["start_frame"], 270)
        self.assertEqual(session.meta["chunks"][3]["end_frame"], 360)
        self.assertEqual(session.meta["chunks"][3]["frame_count"], 90)
        self.assertEqual(session.meta["chunks"][17]["start_frame"], 1530)
        self.assertEqual(session.meta["chunks"][17]["end_frame"], 1608)
        self.assertEqual(session.meta["chunks"][17]["frame_count"], 78)

    def test_incremental_patch_storage_and_streaming_export(self):
        """Verify incremental patch-based storage without full master_frames materialization and streaming export."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            self.skipTest("ffmpeg not available")

        # 1. Create a synthetic 60-frame mp4 video
        video_path = os.path.join(self.test_dir, "synth_patch_test.mp4")
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2.5:size=64x48:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2.5",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.isfile(video_path):
            self.skipTest("ffmpeg synthetic video generation failed")

        p_name = "UnitTest_IncrementalPatch"
        session = get_or_create_timeline_session(p_name)
        session.initialize_from_video_file(
            video_path=video_path,
            force_fps=24.0,
            chunk_length=30,
            target_width=64,
            target_height=48,
        )

        # In streaming mode, master_frames MUST be None initially
        self.assertIsNone(session.master_frames)
        self.assertEqual(len(session.meta["chunks"]), 2)  # 0~30, 30~60

        # 2. Patch Chunk 0: distinctive values (0.88)
        patch0 = torch.full((30, 48, 64, 3), 0.88, dtype=torch.float32)
        session.patch_chunk(
            chunk_index=0,
            start_frame=0,
            end_frame=30,
            edited_images=patch0,
            seam_blend_frames=0,
        )

        # Critical verification: master_frames must STILL be None (no 120GB RAM allocation!)
        self.assertIsNone(session.master_frames)

        # Verify patch was saved individually as safetensors
        self.assertTrue(session.has_chunk_patch(0))
        loaded_p0 = session.load_chunk_patch(0, as_float=True)
        self.assertIsNotNone(loaded_p0)
        self.assertTrue(torch.allclose(loaded_p0["frames"][:5], torch.tensor(0.88), atol=0.02))

        # 3. Verify get_previous_reference_frames for Chunk 1 loads from Chunk 0's patch
        last_f, seq_f, is_edited = session.get_previous_reference_frames(
            start_frame=30,
            ref_frames_count=10,
            target_width=64,
            target_height=48,
        )
        self.assertTrue(is_edited)
        self.assertEqual(last_f.shape, (1, 48, 64, 3))
        self.assertEqual(seq_f.shape, (10, 48, 64, 3))
        self.assertTrue(torch.allclose(seq_f, torch.tensor(0.88), atol=0.02))
        self.assertIsNone(session.master_frames)  # Still zero RAM allocation!

        # 4. Verify streaming MP4 export without master_frames
        from engine.timeline_session_manager import export_master_to_video_file
        export_res = export_master_to_video_file(
            project_name=p_name,
            output_dir=self.test_dir,
            filename="streaming_export_test.mp4",
        )
        self.assertTrue(export_res["success"])
        self.assertTrue(os.path.isfile(export_res["file_path"]))
        self.assertEqual(export_res["total_frames"], 60)

        # 5. Verify reset_chunk deletes the patch
        session.reset_chunk(0)
        self.assertFalse(session.has_chunk_patch(0))
        self.assertIsNone(session.load_chunk_patch(0))


if __name__ == "__main__":
    unittest.main()


