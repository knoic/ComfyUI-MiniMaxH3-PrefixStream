"""Small continuation state carried with sampler latents and persisted AV assets."""

import json
import torch

STATE_KEYS = ("minimax_video_start_step", "minimax_audio_start_tick")
ANCHOR_KEY = "minimax_identity_anchor"
REFERENCES_KEY = "minimax_identity_refs"
MANIFEST_KEY = "minimax_identity_refs_json"


def encode_state(latent):
    tensors = {}
    anchor = latent.get(ANCHOR_KEY)
    refs = latent.get(REFERENCES_KEY) or []
    if anchor is not None and (not refs or anchor is not refs[0]["latent"]):
        tensors[ANCHOR_KEY] = anchor.detach().cpu().contiguous().clone()
    metadata = {key: str(int(latent.get(key, 0))) for key in STATE_KEYS}
    manifest = []
    for index, ref in enumerate(refs):
        key = f"minimax_image_ref_{index}"
        z = ref["latent"]
        tensors[key] = z.detach().cpu().contiguous().clone()
        manifest.append({"tensor": key, "latent_h": ref.get("latent_h", z.shape[-2]),
                         "latent_w": ref.get("latent_w", z.shape[-1])})
    metadata[MANIFEST_KEY] = json.dumps(manifest)
    return tensors, metadata


def decode_state(tensors, metadata):
    state = {key: int(metadata.get(key, 0)) for key in STATE_KEYS}
    if ANCHOR_KEY in tensors:
        state[ANCHOR_KEY] = tensors[ANCHOR_KEY]
    refs = tensors.get(REFERENCES_KEY)
    if refs is None:
        refs = [{"kind": "image", "latent": tensors[item["tensor"]],
                 "latent_h": item["latent_h"], "latent_w": item["latent_w"]}
                for item in json.loads(metadata.get(MANIFEST_KEY, "[]"))]
    if refs:
        state[REFERENCES_KEY] = refs
        state.setdefault(ANCHOR_KEY, refs[0]["latent"])
    return state


def prepare_identity_references(conditioning, previous=None, explicit_anchor=None):
    """Retain native image references without inventing references from generated frames.

    Current native images replace saved images. Saved images fill an omitted
    image reference bank; an explicit identity input may add a new image.
    """
    previous = previous or {}
    current = []
    for _, extra in conditioning or []:
        for ref in extra.get("minimax_refs") or []:
            if ref.get("kind") == "image" and not any(_same_image(ref, old) for old in current):
                current.append(dict(ref))
    refs = current or list(previous.get(REFERENCES_KEY) or [])
    if explicit_anchor is not None:
        ref = {"kind": "image", "latent": explicit_anchor,
               "latent_h": explicit_anchor.shape[-2], "latent_w": explicit_anchor.shape[-1]}
        if not any(_same_image(ref, old) for old in refs):
            refs = refs + [ref]
    state = {REFERENCES_KEY: refs}
    if refs:
        state[ANCHOR_KEY] = refs[0]["latent"]
    if not refs or (current and explicit_anchor is None):
        return conditioning, state
    out = []
    for embedding, extra in conditioning:
        updated = dict(extra)
        existing = list(updated.get("minimax_refs") or [])
        additions = [ref for ref in refs if not any(_same_image(ref, old) for old in existing)]
        if additions:
            updated["minimax_refs"] = ([ref for ref in existing if ref.get("kind") == "image"]
                                        + additions + [ref for ref in existing if ref.get("kind") != "image"])
        out.append([embedding, updated])
    return out, state


def _same_image(left, right):
    if left.get("kind") != "image" or right.get("kind") != "image":
        return False
    a, b = left["latent"], right["latent"]
    return a is b or (a.shape == b.shape and torch.equal(a.to(device=b.device), b))
