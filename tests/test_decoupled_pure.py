"""Unit tests for Decoupled Pure Prefix KV Cache mode (Zero Overlap, Prompt-Aligned)."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    from unittest.mock import MagicMock
    mock_torch = MagicMock()
    mock_torch.float8_e4m3fn = "fp8"
    mock_torch.bfloat16 = "bf16"
    mock_torch.float16 = "fp16"
    mock_torch.float32 = "fp32"
    mock_torch.Tensor = MagicMock
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()


class TestDecoupledPureConfig(unittest.TestCase):
    """Tests configuration and mode detection for Decoupled Pure Prefix."""

    def test_mode_detection(self):
        from engine.cache_manager import KVCacheConfig

        cfg_decoupled = KVCacheConfig(cache_mode="Decoupled Pure Prefix (Zero Overlap, Prompt-Aligned)")
        self.assertTrue(cfg_decoupled.is_decoupled_mode())
        self.assertTrue(cfg_decoupled.is_cache_enabled())

        cfg_safe = KVCacheConfig(cache_mode="Safe Native (Zero Artifacts, Recommended)")
        self.assertFalse(cfg_safe.is_decoupled_mode())
        self.assertFalse(cfg_safe.is_cache_enabled())

        cfg_step1 = KVCacheConfig(cache_mode="Step-1 Dynamic Cache (Experimental Acceleration)")
        self.assertFalse(cfg_step1.is_decoupled_mode())
        self.assertTrue(cfg_step1.is_cache_enabled())

    def test_node_config_creation(self):
        from nodes import MiniMaxPrefixCacheConfigNode

        node = MiniMaxPrefixCacheConfigNode()
        (cfg,) = node.create_config(cache_mode="Decoupled Pure Prefix (Zero Overlap, Prompt-Aligned)")
        self.assertTrue(cfg.is_decoupled_mode())
        self.assertTrue(cfg.is_cache_enabled())


class TestDecoupledRopeAlignment(unittest.TestCase):
    """Tests 3D-RoPE contiguous temporal grid alignment for decoupled mode."""

    def test_spans_continuity_math(self):
        from engine.rope_aligner import video_t_spans, total_span_for_steps

        prefix_steps = 7   # 22 pixel frames
        target_steps = 31  # 124 pixel frames

        prefix_span = total_span_for_steps(prefix_steps)
        self.assertGreater(prefix_span, 0.0)

        prefix_spans = video_t_spans(prefix_steps)
        target_spans = video_t_spans(target_steps)

        self.assertEqual(len(prefix_spans), prefix_steps)
        self.assertEqual(len(target_spans), target_steps)

        # Ensure all steps advance time strictly positive
        self.assertTrue(all(s > 0 for s in prefix_spans))
        self.assertTrue(all(s > 0 for s in target_spans))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for RoPE tensor tests")
    def test_decoupled_coords_continuity(self):
        from engine.rope_aligner import TemporalCursorTracker, total_span_for_steps

        tracker = TemporalCursorTracker()
        prefix_steps = 7   # 22 pixel frames
        target_steps = 31  # 124 pixel frames

        prefix_coords, target_coords = tracker.get_decoupled_coords(
            prefix_steps=prefix_steps,
            target_steps=target_steps,
            base_origin=0.0
        )

        self.assertEqual(prefix_coords.shape[0], prefix_steps)
        self.assertEqual(target_coords.shape[0], target_steps)

        # Prefix should start at 0
        self.assertAlmostEqual(prefix_coords[0].item(), 0.0)

        # Target should start exactly where prefix ended
        expected_target_origin = total_span_for_steps(prefix_steps)
        self.assertAlmostEqual(target_coords[0].item(), expected_target_origin)

        # Target coordinates should be strictly monotonically increasing and positive
        self.assertTrue(torch.all(target_coords >= 0.0))
        diffs = target_coords[1:] - target_coords[:-1]
        self.assertTrue(torch.all(diffs > 0.0))


class TestDecoupledAttention(unittest.TestCase):
    """Tests asymmetric attention operator with pure target Query."""

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for attention tensor tests")
    def test_asymmetric_dimensions(self):
        from engine.fused_attention import asymmetric_cached_attention

        heads = 4
        dim = 32
        s_target = 64
        s_prefix = 16
        s_total = s_prefix + s_target

        q = torch.randn(1, heads, s_target, dim)
        k_cached = torch.randn(1, heads, s_total, dim)
        v_cached = torch.randn(1, heads, s_total, dim)

        out = asymmetric_cached_attention(
            q=q,
            k_cached=k_cached,
            v_cached=v_cached,
            num_heads=heads
        )

        # Output should have length s_target, matching query
        self.assertEqual(out.shape[0], s_target)
        self.assertEqual(out.shape[1], heads * dim)


class TestDecoupledTrimBypass(unittest.TestCase):
    """Tests automatic bypass in Trim node for decoupled mode."""

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for Trim node tests")
    def test_trim_bypass(self):
        from engine.cache_manager import KVCacheConfig
        from pipeline.long_video_director import LongVideoSession
        from nodes import MiniMaxTrimPrefixLatentNode

        cfg = KVCacheConfig(cache_mode="Decoupled Pure Prefix (Zero Overlap, Prompt-Aligned)")
        session = LongVideoSession(cfg)
        session.last_rolling_frames = 22  # previous clip setting

        trim_node = MiniMaxTrimPrefixLatentNode()
        dummy_images = torch.zeros(124, 768, 768, 3)

        # When trim_frames=0 and decoupled mode is active, images must NOT be trimmed
        out_images, _, _ = trim_node.trim(
            trim_frames=0,
            images=dummy_images,
            session=session
        )

        self.assertEqual(out_images.shape[0], 124)


if __name__ == "__main__":
    unittest.main()
