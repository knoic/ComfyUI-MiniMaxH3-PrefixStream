"""Engine package for MiniMax H3 Prefix KV Caching."""

from .cache_manager import PrefixKVCacheManager, KVCacheConfig
from .block_hook import create_prefix_dit_hook
from .warmup_executor import WarmupExecutor
from .rope_aligner import TemporalCursorTracker, video_t_spans, video_t_grid
from .fused_attention import asymmetric_cached_attention

__all__ = [
    "PrefixKVCacheManager",
    "KVCacheConfig",
    "create_prefix_dit_hook",
    "WarmupExecutor",
    "TemporalCursorTracker",
    "video_t_spans",
    "video_t_grid",
    "asymmetric_cached_attention",
]
