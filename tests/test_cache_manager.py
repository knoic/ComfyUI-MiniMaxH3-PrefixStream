"""Unit tests for PrefixKVCacheManager."""

import sys
import os
import unittest
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from engine.cache_manager import (
    PrefixKVCacheManager,
    KVCacheConfig,
    snap_to_run_grid,
    pixel_frames_to_latent_steps,
    latent_steps_to_pixel_frames,
)


class TestCacheManager(unittest.TestCase):
    def test_cache_initialization(self):
        config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
        mgr = PrefixKVCacheManager(config)
        self.assertFalse(mgr.has_cache(0))
        self.assertFalse(mgr.has_cache(49))

    def test_set_and_get_kv(self):
        config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
        mgr = PrefixKVCacheManager(config)

        k_sample = torch.randn(1, 56, 100, 128, dtype=torch.bfloat16)
        v_sample = torch.randn(1, 56, 100, 128, dtype=torch.bfloat16)

        mgr.set_rolling_kv(0, k_sample, v_sample)
        self.assertTrue(mgr.has_cache(0))

        out_k, out_v = mgr.get_combined_kv(0, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
        self.assertEqual(out_k.shape, (1, 56, 100, 128))
        self.assertEqual(out_v.shape, (1, 56, 100, 128))
        self.assertTrue(torch.allclose(out_k, k_sample))

    def test_anchor_plus_rolling_concatenation(self):
        config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
        mgr = PrefixKVCacheManager(config)

        k_anc = torch.randn(1, 56, 30, 128, dtype=torch.bfloat16)
        v_anc = torch.randn(1, 56, 30, 128, dtype=torch.bfloat16)
        k_rol = torch.randn(1, 56, 70, 128, dtype=torch.bfloat16)
        v_rol = torch.randn(1, 56, 70, 128, dtype=torch.bfloat16)

        mgr.set_anchor_kv(5, k_anc, v_anc)
        mgr.set_rolling_kv(5, k_rol, v_rol)

        out_k, out_v = mgr.get_combined_kv(5, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
        self.assertEqual(out_k.shape, (1, 56, 100, 128))
        self.assertEqual(out_v.shape, (1, 56, 100, 128))

    def test_prefetch_slot_mechanism(self):
        config = KVCacheConfig(num_layers=10, num_heads=4, head_dim=32, cache_dtype="bf16", device_mode="cpu_pinned")
        mgr = PrefixKVCacheManager(config)

        k = torch.randn(1, 4, 20, 32, dtype=torch.bfloat16)
        v = torch.randn(1, 4, 20, 32, dtype=torch.bfloat16)
        mgr.set_rolling_kv(1, k, v)

        if torch.cuda.is_available():
            mgr.prefetch_next_layer(1, torch.device("cuda"))
            self.assertIn(1, mgr._prefetch_slot)
        else:
            mgr._prefetch_slot[1] = (None, None, k, v)
            self.assertIn(1, mgr._prefetch_slot)

        out_k, out_v = mgr.get_combined_kv(1, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
        self.assertEqual(out_k.shape, (1, 4, 20, 32))
        self.assertNotIn(1, mgr._prefetch_slot)

    def test_memory_tracking(self):
        config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64, cache_dtype="fp16", device_mode="gpu")
        mgr = PrefixKVCacheManager(config)

        k = torch.randn(1, 8, 50, 64, dtype=torch.float16)
        v = torch.randn(1, 8, 50, 64, dtype=torch.float16)
        mgr.set_rolling_kv(0, k, v)

        mem = mgr.get_memory_usage_mb()
        self.assertGreater(mem["total_mb"], 0)

    def test_grid_math_utilities(self):
        self.assertEqual(snap_to_run_grid(39), 39)
        self.assertEqual(snap_to_run_grid(45), 39)
        self.assertEqual(snap_to_run_grid(90), 90)
        self.assertEqual(pixel_frames_to_latent_steps(39), 12)
        self.assertEqual(latent_steps_to_pixel_frames(12), 39)


if __name__ == "__main__":
    unittest.main()
