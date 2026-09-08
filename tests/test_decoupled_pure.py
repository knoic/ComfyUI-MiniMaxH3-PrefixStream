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


class TestDecoupledWarmupAndHookEndToEnd(unittest.TestCase):
    """End-to-end integration test for Phase 0 warmup and Phase 1 decoupled hook execution."""

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for integration tests")
    def test_warmup_and_hook_execution(self):
        import torch.nn as nn
        from engine.cache_manager import PrefixKVCacheManager, KVCacheConfig
        from engine.warmup_executor import WarmupExecutor
        from engine.block_hook import create_prefix_dit_hook

        heads = 4
        head_dim = 32
        hidden = heads * head_dim
        num_layers = 4  # fast test

        # 1. Build a mock DiT model
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
                self.q_norm = nn.LayerNorm(head_dim)
                self.k_norm = nn.LayerNorm(head_dim)
                self.out_proj = nn.Linear(hidden, hidden, bias=False)

            def forward(self, x, rope_freqs=None, transformer_options=None):
                q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
                return self.out_proj(q)

        class MockBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MockAttention()

            def forward(self, x, t_emb=None, mod_segments=None, rope_freqs=None, transformer_options=None, attention=None):
                attn_fn = self.attn if attention is None else attention
                return x + attn_fn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

        class MockDiffusionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockBlock() for _ in range(num_layers)])
                self.patch_size = (1, 2, 2)
                self.video_patch_proj = nn.Linear(16 * 1 * 2 * 2, hidden)
                self.audio_patch_proj = nn.Linear(32, hidden)
                self.text_dim = 5120

            def forward(self, x, timestep, context, transformer_options=None):
                # Simple forward pass that triggers patches_replace hooks
                patches_replace = (transformer_options or {}).get("patches_replace", {})
                dit_patches = patches_replace.get("dit", {})

                # Dummy sequence
                h = torch.randn(30, hidden)
                for i, block in enumerate(self.blocks):
                    if ("double_block", i) in dit_patches:
                        def block_wrap(args):
                            return {"img": block(args["img"], transformer_options=args.get("transformer_options"), attention=args.get("attention"))}
                        h = dit_patches[("double_block", i)](
                            {"img": h, "transformer_options": transformer_options},
                            {"original_block": block_wrap}
                        )["img"]
                    else:
                        h = block(h)
                return [x[0], x[1]]

        mock_diff = MockDiffusionModel()
        mock_patcher = type("MockPatcher", (), {"model": type("M", (), {"diffusion_model": mock_diff})()})()

        # 2. Phase 0: Run WarmupExecutor
        cfg = KVCacheConfig(cache_mode="Decoupled Pure Prefix (Zero Overlap, Prompt-Aligned)", num_layers=num_layers, num_heads=heads, head_dim=head_dim)
        cache_mgr = PrefixKVCacheManager(cfg)
        warmup_exec = WarmupExecutor(cache_mgr)

        v_prefix = torch.randn(1, 16, 7, 16, 16)
        a_prefix = torch.randn(1, 32, 2, 11)
        text_ctx = torch.randn(1, 5, 5120)

        success = warmup_exec.precompute_rolling(
            model_patcher=mock_patcher,
            prefix_video_latent=v_prefix,
            prefix_audio_latent=a_prefix,
            text_context=text_ctx
        )
        self.assertTrue(success)
        self.assertTrue(cache_mgr.has_cache(0))
        self.assertTrue(cache_mgr.has_cache(num_layers - 1))

        # 3. Phase 1: Run Decoupled Pure hook on target generation step
        decoupled_hooks = create_prefix_dit_hook(cache_mgr, model=mock_diff)
        target_s = 64
        h_target = torch.randn(target_s, hidden)

        hook_fn = decoupled_hooks[("double_block", 0)]
        block_wrap_called = False

        def mock_wrap(args):
            nonlocal block_wrap_called
            block_wrap_called = True
            # Simulate block calling the attention function passed in args
            attn_fn = args.get("attention")
            self.assertIsNotNone(attn_fn)
            out_attn = attn_fn(args["img"])
            self.assertEqual(out_attn.shape, args["img"].shape)
            return {"img": args["img"] + out_attn}

        res = hook_fn(
            {"img": h_target, "transformer_options": {"minimax_prefix_mode": "decoupled_pure"}},
            {"original_block": mock_wrap}
        )

        self.assertTrue(block_wrap_called)
        self.assertEqual(res["img"].shape, (target_s, hidden))
        self.assertEqual(cache_mgr.step_counter, 1)


if __name__ == "__main__":
    unittest.main()

