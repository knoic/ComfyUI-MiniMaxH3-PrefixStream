"""Unit tests for PrefixKVCacheManager."""

import sys
import os
import torch

# Ensure parent directory is in sys.path for local standalone testing
_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

try:
    from engine.cache_manager import PrefixKVCacheManager, KVCacheConfig
except ImportError:
    from ComfyUI_MiniMaxH3_PrefixStream.engine.cache_manager import PrefixKVCacheManager, KVCacheConfig


def test_cache_initialization():
    config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
    mgr = PrefixKVCacheManager(config)
    assert not mgr.has_cache(0)
    assert not mgr.has_cache(49)


def test_set_and_get_kv():
    config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
    mgr = PrefixKVCacheManager(config)

    # Simulate Layer 0: [1, 56, 100, 128]
    k_sample = torch.randn(1, 56, 100, 128, dtype=torch.bfloat16)
    v_sample = torch.randn(1, 56, 100, 128, dtype=torch.bfloat16)

    mgr.set_rolling_kv(0, k_sample, v_sample)
    assert mgr.has_cache(0)

    out_k, out_v = mgr.get_combined_kv(0, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
    assert out_k.shape == (1, 56, 100, 128)
    assert out_v.shape == (1, 56, 100, 128)
    assert torch.allclose(out_k, k_sample)


def test_anchor_plus_rolling_concatenation():
    config = KVCacheConfig(num_layers=50, num_heads=56, head_dim=128, cache_dtype="bf16", device_mode="gpu")
    mgr = PrefixKVCacheManager(config)

    # Anchor: 30 tokens, Rolling: 70 tokens
    k_anc = torch.randn(1, 56, 30, 128, dtype=torch.bfloat16)
    v_anc = torch.randn(1, 56, 30, 128, dtype=torch.bfloat16)
    k_rol = torch.randn(1, 56, 70, 128, dtype=torch.bfloat16)
    v_rol = torch.randn(1, 56, 70, 128, dtype=torch.bfloat16)

    mgr.set_anchor_kv(5, k_anc, v_anc)
    mgr.set_rolling_kv(5, k_rol, v_rol)

    out_k, out_v = mgr.get_combined_kv(5, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
    # Total sequence length should be 30 + 70 = 100 tokens
    assert out_k.shape == (1, 56, 100, 128)
    assert out_v.shape == (1, 56, 100, 128)


def test_prefetch_slot_mechanism():
    config = KVCacheConfig(num_layers=10, num_heads=4, head_dim=32, cache_dtype="bf16", device_mode="cpu_pinned")
    mgr = PrefixKVCacheManager(config)

    k = torch.randn(1, 4, 20, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 20, 32, dtype=torch.bfloat16)
    mgr.set_rolling_kv(1, k, v)

    if torch.cuda.is_available():
        mgr.prefetch_next_layer(1, torch.device("cuda"))
        assert 1 in mgr._prefetch_slot
    else:
        # Mock prefetch slot insertion to test consumption and purge logic on CPU
        mgr._prefetch_slot[1] = (None, None, k, v)
        assert 1 in mgr._prefetch_slot
    
    # Combined KV should consume from prefetch slot
    out_k, out_v = mgr.get_combined_kv(1, target_device=torch.device("cpu"), compute_dtype=torch.bfloat16)
    assert out_k.shape == (1, 4, 20, 32)
    assert 1 not in mgr._prefetch_slot  # Consumed and purged


def test_memory_tracking():
    config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64, cache_dtype="fp16", device_mode="gpu")
    mgr = PrefixKVCacheManager(config)

    k = torch.randn(1, 8, 50, 64, dtype=torch.float16)
    v = torch.randn(1, 8, 50, 64, dtype=torch.float16)
    mgr.set_rolling_kv(0, k, v)

    mem = mgr.get_memory_usage_mb()
    assert mem["total_mb"] > 0


if __name__ == "__main__":
    test_cache_initialization()
    test_set_and_get_kv()
    test_anchor_plus_rolling_concatenation()
    test_prefetch_slot_mechanism()
    test_memory_tracking()
    print("All cache_manager tests passed successfully!")
