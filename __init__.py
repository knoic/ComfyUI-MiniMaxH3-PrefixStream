"""ComfyUI-MiniMaxH3-PrefixStream

A high-performance Prefix KV Caching & Streaming Chaining Suite for MiniMax H3.
Slashes DiT computation by ~45% and prevents long-video degradation via Dual-Tier
(Anchor + Rolling) Attention Caching.
"""

from .nodes import (
    MiniMaxPrefixCacheConfigNode,
    MiniMaxPrefixCacheApplierNode,
    MiniMaxLongVideoStitcherNode,
    MiniMaxCacheMonitorNode,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxLongVideoStitcher": MiniMaxLongVideoStitcherNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Prefix Cache Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Prefix Cache Applier",
    "MiniMaxLongVideoStitcher": "MiniMax H3 Long Video Stitcher",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
