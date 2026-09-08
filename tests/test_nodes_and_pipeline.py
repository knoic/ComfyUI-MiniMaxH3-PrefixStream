"""Integration unit tests for ComfyUI custom nodes and pipeline."""

import sys
import os
import shutil
import tempfile
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

import __init__ as pkg_init
import nodes


def test_node_mappings_consistency():
    """Verify all defined nodes in nodes.py are exported in __init__.py."""
    assert set(pkg_init.NODE_CLASS_MAPPINGS.keys()) == set(nodes.NODE_CLASS_MAPPINGS.keys())
    assert set(pkg_init.NODE_DISPLAY_NAME_MAPPINGS.keys()) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS.keys())
    assert "MiniMaxSaveLatent" in pkg_init.NODE_CLASS_MAPPINGS
    assert "MiniMaxLoadLatent" in pkg_init.NODE_CLASS_MAPPINGS
    assert "MiniMaxTrimPrefix" in pkg_init.NODE_CLASS_MAPPINGS


def test_config_node():
    cfg_node = nodes.MiniMaxPrefixCacheConfigNode()
    (cfg,) = cfg_node.create_config(
        cache_mode="Safe Native (Zero Artifacts, Recommended)",
        cache_dtype="fp8",
        device_mode="auto",
        rolling_frames="22",
        use_anchor=False,
        anchor_frames=5
    )
    assert cfg.rolling_frames == 22
    assert cfg.rolling_latent_frames == 7
    assert not cfg.is_cache_enabled()


def test_av_latent_pack_unpack():
    v = torch.randn(1, 24, 10, 16, 16)
    a = torch.randn(1, 32, 2, 16)

    latent_dict = nodes.pack_av_latent(v, a)
    unpacked_v, unpacked_a = nodes._unpack_latent(latent_dict)

    assert unpacked_v is not None and unpacked_v.shape == v.shape
    assert unpacked_a is not None and unpacked_a.shape == a.shape


def test_save_and_load_latent_node():
    temp_dir = tempfile.mkdtemp(prefix="minimax_test_")
    try:
        # Override folder_paths to return temp_dir
        import types
        dummy_folder_paths = types.ModuleType("folder_paths")
        dummy_folder_paths.get_output_directory = lambda: temp_dir
        sys.modules["folder_paths"] = dummy_folder_paths

        v = torch.randn(1, 24, 7, 16, 16)
        a = torch.randn(1, 32, 2, 11)
        original_latent = nodes.pack_av_latent(v, a)

        save_node = nodes.MiniMaxSaveLatentNode()
        saved_path, info = save_node.save(
            latent=original_latent,
            filename_prefix="test_clip/seg",
            clip_index=1
        )
        assert os.path.exists(saved_path)
        assert "frames" in info

        load_node = nodes.MiniMaxLoadLatentNode()
        loaded_latent, loaded_path, load_info = load_node.load(
            latent_path=saved_path,
            clip_index=1
        )
        assert loaded_path == saved_path
        lv, la = nodes._unpack_latent(loaded_latent)
        assert lv.shape == v.shape
        if la is not None:
            assert la.shape == a.shape
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trim_node_pixel_and_waveform():
    trim_node = nodes.MiniMaxTrimPrefixLatentNode()
    images = torch.ones((48, 32, 32, 3), dtype=torch.float32)
    audio = {"waveform": torch.ones((1, 2, 64000)), "sample_rate": 32000}

    out_img, out_aud, _ = trim_node.trim(
        trim_frames=24,
        images=images,
        audio=audio,
        fps=24.0
    )
    assert out_img.shape[0] == 24
    assert out_aud["waveform"].shape[-1] == 32000


def test_workflow_graph_integrity():
    import json
    wf_file = os.path.join(_repo_dir, "examples", "MiniMaxH3_PrefixStream_LongVideo_Workflow.json")
    if not os.path.exists(wf_file):
        return
    with open(wf_file, "r", encoding="utf-8") as f:
        wf = json.load(f)
    nodes_dict = {n["id"]: n for n in wf.get("nodes", [])}
    links_dict = {l[0]: l for l in wf.get("links", []) if l}
    for nid, n in nodes_dict.items():
        for slot, inp in enumerate(n.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None:
                assert lid in links_dict, f"Link {lid} on Node {nid} slot {slot} missing from global links!"
                l = links_dict[lid]
                assert l[3] == nid and l[4] == slot, f"Link {lid} target mismatch!"


if __name__ == "__main__":
    test_node_mappings_consistency()
    test_config_node()
    test_av_latent_pack_unpack()
    test_save_and_load_latent_node()
    test_trim_node_pixel_and_waveform()
    test_workflow_graph_integrity()
    print("All nodes and pipeline integration tests passed successfully!")
