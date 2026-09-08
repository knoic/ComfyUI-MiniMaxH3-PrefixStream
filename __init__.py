"""ComfyUI-MiniMaxH3-PrefixStream

A high-performance Prefix KV Caching & Streaming Chaining Suite for MiniMax H3.
Slashes DiT computation and prevents long-video degradation via Dual-Tier
(Anchor + Rolling) Attention Caching and Seamless AV Handover.
"""

try:
    from .nodes import (
        MiniMaxPrefixCacheConfigNode,
        MiniMaxPrefixCacheApplierNode,
        MiniMaxTrimPrefixLatentNode,
        MiniMaxLongVideoStitcherNode,
        MiniMaxCacheMonitorNode,
        MiniMaxSaveLatentNode,
        MiniMaxLoadLatentNode,
        MiniMaxClipBinSaverNode,
        MiniMaxClipBinPickerNode,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except (ImportError, ValueError):
    from nodes import (
        MiniMaxPrefixCacheConfigNode,
        MiniMaxPrefixCacheApplierNode,
        MiniMaxTrimPrefixLatentNode,
        MiniMaxLongVideoStitcherNode,
        MiniMaxCacheMonitorNode,
        MiniMaxSaveLatentNode,
        MiniMaxLoadLatentNode,
        MiniMaxClipBinSaverNode,
        MiniMaxClipBinPickerNode,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

__all__ = [
    "MiniMaxPrefixCacheConfigNode",
    "MiniMaxPrefixCacheApplierNode",
    "MiniMaxTrimPrefixLatentNode",
    "MiniMaxLongVideoStitcherNode",
    "MiniMaxCacheMonitorNode",
    "MiniMaxSaveLatentNode",
    "MiniMaxLoadLatentNode",
    "MiniMaxClipBinSaverNode",
    "MiniMaxClipBinPickerNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

