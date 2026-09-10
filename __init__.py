"""ComfyUI MiniMax H3 native masked AV continuation and streaming suite."""

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
    from .engine.clip_bin_api import register_clip_bin_routes
    register_clip_bin_routes()
except Exception:
    try:
        from engine.clip_bin_api import register_clip_bin_routes
        register_clip_bin_routes()
    except Exception:
        pass

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
