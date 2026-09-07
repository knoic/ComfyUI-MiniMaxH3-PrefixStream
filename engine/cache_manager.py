"""Prefix KV Cache Manager for MiniMax H3 (50-layer DiT).

Provides memory-efficient storage, quantization (FP8/BF16), dual-tier caching
(Anchor + Rolling), and asynchronous CPU-pinned memory streaming.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging
import torch

logger = logging.getLogger("minimax_prefix_stream")


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5)


def snap_to_run_grid(frames: int) -> int:
    """Snaps a requested frame count down to the nearest MiniMax H3 VAE grid point."""
    if frames <= 0:
        return 0
    return next((g for g in VIDEO_RUN_GRID if g <= frames), 5)


def steps_for_frames(frames: int) -> Optional[int]:
    """Returns the exact number of latent steps covering frames from cycle position 0, or None if off-grid."""
    k, covered = 0, 0
    while covered < frames:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == frames else None


def pixel_frames_to_latent_steps(pixel_frames: int) -> int:
    """Converts pixel frames to the minimum MiniMax H3 latent steps covering them."""
    if pixel_frames <= 0:
        return 0
    k, covered = 0, 0
    while covered < pixel_frames:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return max(1, k)


def latent_steps_to_pixel_frames(latent_steps: int) -> int:
    """Calculates exact pixel frames spanned by MiniMax H3 latent steps."""
    if latent_steps <= 0:
        return 0
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_steps))


@dataclass
class KVCacheConfig:
    """Configuration for MiniMax H3 Prefix KV Cache."""
    cache_mode: str = "Safe Native (Zero Artifacts, Recommended)"
    num_layers: int = 50
    num_heads: int = 56
    head_dim: int = 128
    cache_dtype: str = "fp8"          # "fp8", "bf16", "fp16"
    device_mode: str = "auto"         # "gpu", "cpu_pinned", "auto"
    use_anchor: bool = False          # False avoids redundant anchor collision when rolling tail is present
    rolling_frames: int = 22          # Real video frames for rolling window (default 22 frames = 7 latent steps)
    anchor_frames: int = 5            # Real video frames for anchor (default 5 frames = 2 latent steps)
    temporal_stride: int = 1          # 1 = keep all, 2 = 2x temporal sub-sampling

    # Internal overrides / compatibility
    _rolling_latent_frames: Optional[int] = None
    _anchor_latent_frames: Optional[int] = None

    def is_cache_enabled(self) -> bool:
        """Returns True if DiT KV caching is enabled; False for 100% native ComfyUI attention."""
        cm = str(self.cache_mode).lower()
        return "step" in cm or "dynamic" in cm or "experimental" in cm

    @property
    def rolling_latent_frames(self) -> int:
        if self._rolling_latent_frames is not None:
            return self._rolling_latent_frames
        snapped = snap_to_run_grid(self.rolling_frames)
        return pixel_frames_to_latent_steps(snapped)

    @rolling_latent_frames.setter
    def rolling_latent_frames(self, val: int):
        self._rolling_latent_frames = val

    @property
    def anchor_latent_frames(self) -> int:
        if self._anchor_latent_frames is not None:
            return self._anchor_latent_frames
        return pixel_frames_to_latent_steps(self.anchor_frames)

    @anchor_latent_frames.setter
    def anchor_latent_frames(self, val: int):
        self._anchor_latent_frames = val

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.cache_dtype.lower() == "fp8":
            if hasattr(torch, "float8_e4m3fn"):
                return torch.float8_e4m3fn
            logger.warning("torch.float8_e4m3fn not supported on this PyTorch version; falling back to bfloat16")
            return torch.bfloat16
        if self.cache_dtype.lower() == "bf16":
            return torch.bfloat16
        return torch.float16


class PrefixKVCacheManager:
    """Manages 50-layer Key and Value tensors for MiniMax H3 across diffusion steps."""

    def __init__(self, config: Optional[KVCacheConfig] = None):
        self.config = config or KVCacheConfig()
        self._cache_dtype = self.config.torch_dtype

        # Storage: layer_idx -> Tensor
        # Shape per layer: [1, num_heads, S_tokens, head_dim]
        self._anchor_k: List[Optional[torch.Tensor]] = [None] * self.config.num_layers
        self._anchor_v: List[Optional[torch.Tensor]] = [None] * self.config.num_layers
        self._rolling_k: List[Optional[torch.Tensor]] = [None] * self.config.num_layers
        self._rolling_v: List[Optional[torch.Tensor]] = [None] * self.config.num_layers

        # CPU-pinned streaming support
        self._stream: Optional[torch.cuda.Stream] = None
        self._prefetch_slot: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._resolved_device_mode = self._resolve_device_mode()

        # Step-1 Dynamic Capture & Runtime Metrics
        self.step_counter: int = 0
        self.is_capturing: bool = False
        self.captured_tokens: int = 0
        self.skipped_steps_count: int = 0
        self.enabled: bool = True

    def _resolve_device_mode(self) -> str:
        mode = self.config.device_mode.lower()
        if mode in ("gpu", "cpu_pinned"):
            return mode
        # Auto mode: check available CUDA VRAM
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                # If free VRAM < 16GB, use cpu_pinned to guarantee zero OOM
                if free_gb < 16.0:
                    logger.info("Auto device_mode: Free VRAM %.2f GB < 16 GB, using cpu_pinned streaming", free_gb)
                    return "cpu_pinned"
                logger.info("Auto device_mode: Free VRAM %.2f GB >= 16 GB, using GPU resident cache", free_gb)
                return "gpu"
            except Exception:
                pass
        return "gpu"

    @property
    def is_cpu_pinned(self) -> bool:
        return self._resolved_device_mode == "cpu_pinned"

    def _prepare_tensor_storage(self, t: torch.Tensor) -> torch.Tensor:
        """Cast tensor to configured cache dtype and target device (GPU or CPU pinned)."""
        t = t.contiguous()

        # Quantize to cache dtype if needed
        if t.dtype != self._cache_dtype:
            t = t.to(self._cache_dtype)

        if self.is_cpu_pinned:
            t = t.cpu()
            if not t.is_pinned():
                t = t.pin_memory()
        return t

    def set_anchor_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store Anchor Key/Value for a specific layer."""
        if not (0 <= layer_idx < self.config.num_layers):
            raise IndexError(f"Layer index {layer_idx} out of range (0..{self.config.num_layers-1})")
        self._anchor_k[layer_idx] = self._prepare_tensor_storage(k)
        self._anchor_v[layer_idx] = self._prepare_tensor_storage(v)

    def set_rolling_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store Rolling Key/Value for a specific layer."""
        if not (0 <= layer_idx < self.config.num_layers):
            raise IndexError(f"Layer index {layer_idx} out of range (0..{self.config.num_layers-1})")
        self._rolling_k[layer_idx] = self._prepare_tensor_storage(k)
        self._rolling_v[layer_idx] = self._prepare_tensor_storage(v)

    def has_cache(self, layer_idx: int = 0) -> bool:
        """Check if cache exists for the specified layer."""
        has_anc = self._anchor_k[layer_idx] is not None
        has_rol = self._rolling_k[layer_idx] is not None
        return has_anc or has_rol

    def get_combined_kv(
        self,
        layer_idx: int,
        target_device: torch.device,
        compute_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch and concatenate Anchor + Rolling KV for layer_idx, cast to compute_dtype on target_device."""
        k_parts = []
        v_parts = []

        # 1. Anchor KV
        ak = self._anchor_k[layer_idx]
        av = self._anchor_v[layer_idx]
        if ak is not None and av is not None:
            if ak.device != target_device:
                ak = ak.to(target_device, non_blocking=True)
                av = av.to(target_device, non_blocking=True)
            if ak.dtype != compute_dtype:
                ak = ak.to(compute_dtype)
                av = av.to(compute_dtype)
            k_parts.append(ak)
            v_parts.append(av)

        # 2. Rolling KV
        rk = self._rolling_k[layer_idx]
        rv = self._rolling_v[layer_idx]
        if rk is not None and rv is not None:
            if rk.device != target_device:
                rk = rk.to(target_device, non_blocking=True)
                rv = rv.to(target_device, non_blocking=True)
            if rk.dtype != compute_dtype:
                rk = rk.to(compute_dtype)
                rv = rv.to(compute_dtype)
            k_parts.append(rk)
            v_parts.append(rv)

        if not k_parts:
            raise RuntimeError(f"No KV cache available for layer {layer_idx}")

        if len(k_parts) == 1:
            return k_parts[0], v_parts[0]

        # Concatenate along token sequence dimension:
        # MiniMax attention operates on [1, heads, S, head_dim]
        cat_dim = 2 if k_parts[0].ndim == 4 and k_parts[0].shape[1] == self.config.num_heads else 1
        comb_k = torch.cat(k_parts, dim=cat_dim)
        comb_v = torch.cat(v_parts, dim=cat_dim)
        return comb_k, comb_v

    def prefetch_next_layer(self, next_layer_idx: int, target_device: torch.device) -> None:
        """Asynchronously prefetch the next layer's KV from CPU pinned memory to GPU."""
        if not self.is_cpu_pinned or next_layer_idx >= self.config.num_layers:
            return
        if self._stream is None and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                ak = self._anchor_k[next_layer_idx]
                av = self._anchor_v[next_layer_idx]
                rk = self._rolling_k[next_layer_idx]
                rv = self._rolling_v[next_layer_idx]
                if ak is not None:
                    ak.to(target_device, non_blocking=True)
                if av is not None:
                    av.to(target_device, non_blocking=True)
                if rk is not None:
                    rk.to(target_device, non_blocking=True)
                if rv is not None:
                    rv.to(target_device, non_blocking=True)

    def clear_rolling(self) -> None:
        """Clear only rolling cache (e.g. at scene cut / jump)."""
        self._rolling_k = [None] * self.config.num_layers
        self._rolling_v = [None] * self.config.num_layers

    def reset_step_counter(self) -> None:
        """Reset step counter for a new sampling run."""
        self.step_counter = 0
        self.is_capturing = False

    def reset_for_next_clip(self) -> None:
        """Reset capture state and rolling cache when moving to the next clip."""
        self.reset_step_counter()
        self.clear_rolling()

    def clear_all(self) -> None:
        """Clear both anchor and rolling caches."""
        self.reset_step_counter()
        self.captured_tokens = 0
        self.skipped_steps_count = 0
        self._anchor_k = [None] * self.config.num_layers
        self._anchor_v = [None] * self.config.num_layers
        self.clear_rolling()

    def get_memory_usage_mb(self) -> Dict[str, float]:
        """Calculate memory footprint in MB."""
        gpu_bytes = 0
        cpu_bytes = 0
        for tensors in (self._anchor_k, self._anchor_v, self._rolling_k, self._rolling_v):
            for t in tensors:
                if t is not None:
                    nbytes = t.numel() * t.element_size()
                    if t.is_cuda:
                        gpu_bytes += nbytes
                    else:
                        cpu_bytes += nbytes
        return {
            "gpu_mb": gpu_bytes / (1024 * 1024),
            "cpu_pinned_mb": cpu_bytes / (1024 * 1024),
            "total_mb": (gpu_bytes + cpu_bytes) / (1024 * 1024),
        }
