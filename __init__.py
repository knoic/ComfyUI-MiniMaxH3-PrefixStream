"""ComfyUI MiniMax H3 native masked AV continuation and streaming suite."""

import logging

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
        MiniMaxSafeVAEDecodeNode,
        MiniMaxSafeVAEDecodeAudioNode,
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
        MiniMaxSafeVAEDecodeNode,
        MiniMaxSafeVAEDecodeAudioNode,
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

try:
    if __package__:
        from .engine.clip_bin_api import register_clip_bin_routes
    else:
        from engine.clip_bin_api import register_clip_bin_routes
    register_clip_bin_routes()
except Exception:
    logging.getLogger("minimax_prefix_stream").exception("Clip Bin route registration failed")

WEB_DIRECTORY = "./web"

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
    "MiniMaxSafeVAEDecodeNode",
    "MiniMaxSafeVAEDecodeAudioNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
