# Copyright © 2026 Apple Inc.

"""Save and load the server's prompt cache on local disk.

The server holds prefilled KV caches in an ``LRUPromptCache``. A long prompt
costs minutes of prefill and the cache dies with the process. This module
writes all the entries of one cache to a safetensors file and reads them back.

In a distributed server each rank holds its own shard of the cache and writes
its own file, ``<name>-rank<N>.safetensors``.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .models import cache as cache_module
from .models.cache import LRUPromptCache, make_prompt_cache

VERSION = "1"

DEFAULT_DIR = Path.home() / ".cache" / "mlx-lm" / "prompt-cache"

FINGERPRINT_FIELDS = ("version", "model", "rank", "world", "layers")

_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)

# The tokens of a cache and the offset of its KV layers must agree. Allow a
# small difference because a cache can hold a few tokens more than its key.
_OFFSET_TOLERANCE = 8


def cache_dir(root: Optional[str] = None) -> Path:
    """Return the directory that holds the saved caches."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("MLX_LM_PROMPT_CACHE_DIR", DEFAULT_DIR))


def cache_path(name: str, rank: int, root: Optional[str] = None) -> Path:
    """Return the file that holds the shard of one rank."""
    return cache_dir(root) / f"{name}-rank{rank}.safetensors"


def check_name(name: str) -> str:
    """Refuse a name that can escape the cache directory."""
    if (
        not name
        or len(name) > 128
        or name.startswith(".")
        or not set(name) <= _NAME_CHARS
    ):
        raise ValueError(
            "Invalid cache name. Use 1 to 128 letters, digits, '.', '_' or "
            f"'-' and do not start with '.': {name!r}"
        )
    return name


def layer_kinds(prompt_cache: List[Any]) -> str:
    """Return a signature of the cache type of each layer.

    A quantized KV cache counts as a KV cache. A layer can quantize its cache
    while the server runs, and a quantized cache holds its own group size and
    bits.
    """
    return ",".join(type(c).__name__.replace("Quantized", "") for c in prompt_cache)


def make_fingerprint(model, model_key: Any, rank: int, world: int) -> Dict[str, str]:
    """Return the configuration that a file must match to load.

    A cache from another model or from another number of ranks is not stale,
    it is wrong. Every field must be equal.
    """
    return {
        "version": VERSION,
        "model": repr(model_key),
        "rank": str(rank),
        "world": str(world),
        "layers": layer_kinds(make_prompt_cache(model)),
    }


def saved_names(rank: int, root: Optional[str] = None) -> List[str]:
    """Return the names of the caches that this rank can load."""
    directory = cache_dir(root)
    if not directory.is_dir():
        return []
    suffix = f"-rank{rank}.safetensors"
    return sorted(
        p.name[: -len(suffix)] for p in directory.glob(f"*{suffix}") if p.is_file()
    )


def listing(
    prompt_cache: LRUPromptCache, rank: int, root: Optional[str] = None
) -> Dict[str, Any]:
    """Return the saved caches and the live cache of this rank."""
    return {
        "dir": str(cache_dir(root)),
        "rank": rank,
        "saved": saved_names(rank, root),
        "in_memory": {
            "sequences": len(prompt_cache),
            "bytes": prompt_cache.nbytes,
            "by_type": prompt_cache.stats_by_type(),
        },
    }


def save(
    prompt_cache: LRUPromptCache,
    model_key: Any,
    fingerprint: Dict[str, str],
    path: Path,
    types: Optional[List[str]] = None,
    min_tokens: int = 0,
) -> Dict[str, Any]:
    """Write the entries of ``prompt_cache`` to ``path``.

    Args:
        prompt_cache (LRUPromptCache): The cache to save.
        model_key (Any): Only the entries of this model are saved.
        fingerprint (Dict[str, str]): The output of ``make_fingerprint``.
        path (Path): The ``.safetensors`` file to write.
        types (List[str], optional): Only save these cache types.
        min_tokens (int): Only save entries with at least this many tokens.

    Returns:
        Dict[str, Any]: A report of what the file holds.
    """
    entries = []
    skipped = 0
    n_bytes = 0
    for model, tokens, entry in prompt_cache.items():
        keep = (
            model == model_key
            and len(tokens) >= max(min_tokens, 1)
            and (types is None or entry.cache_type in types)
            and not any(c.empty() for c in entry.prompt_cache)
        )
        if not keep:
            skipped += 1
            continue
        entries.append((list(tokens), entry))
        n_bytes += entry.nbytes

    if not entries:
        raise ValueError(
            f"The prompt cache has nothing to save. It skipped {skipped} "
            f"entries for types={types} and min_tokens={min_tokens}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = int(n_bytes * 1.1) + (1 << 26)
    free = shutil.disk_usage(path.parent).free
    if free < needed:
        raise OSError(
            f"Not enough space in {path.parent}. The save needs about "
            f"{needed / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free."
        )

    # The tokens go in the data. In the metadata a long key becomes a very
    # large JSON string in the file header.
    arrays = {
        "cache": [[c.state for c in e.prompt_cache] for _, e in entries],
        "tokens": [mx.array(t, dtype=mx.int32) for t, _ in entries],
    }
    metadata = {
        "meta": [[c.meta_state for c in e.prompt_cache] for _, e in entries],
        "classes": [[type(c).__name__ for c in e.prompt_cache] for _, e in entries],
        "types": [e.cache_type for _, e in entries],
        "counts": [str(len(t)) for t, _ in entries],
        "fingerprint": fingerprint,
    }
    arrays = dict(tree_flatten(arrays))
    metadata = dict(tree_flatten(metadata))

    # Write to a temporary file in the same directory and rename it, so that a
    # failed save cannot replace a good file with a partial one.
    # mx.save_safetensors adds the extension, so the name must keep it.
    tmp = path.parent / f"{path.stem}.tmp{os.getpid()}.safetensors"
    try:
        mx.save_safetensors(str(tmp), arrays, metadata)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        del arrays
        mx.clear_cache()

    return {
        "entries": len(entries),
        "skipped": skipped,
        "bytes": path.stat().st_size,
        "tokens": [len(t) for t, _ in entries],
        "types": [e.cache_type for _, e in entries],
        "path": str(path),
    }


def load(
    prompt_cache: LRUPromptCache,
    model_key: Any,
    fingerprint: Dict[str, str],
    path: Path,
) -> Dict[str, Any]:
    """Add the entries in ``path`` to ``prompt_cache``.

    Args:
        prompt_cache (LRUPromptCache): The cache to add the entries to.
        model_key (Any): The key to file the entries under.
        fingerprint (Dict[str, str]): The output of ``make_fingerprint``. The
            file must hold the same fingerprint.
        path (Path): The ``.safetensors`` file to read.

    Returns:
        Dict[str, Any]: A report of what the file held.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"There is no saved prompt cache at {path}.")
    arrays, raw_metadata = mx.load(str(path), return_metadata=True)
    data = tree_unflatten(list(arrays.items()))
    metadata = tree_unflatten(list(raw_metadata.items()))
    del arrays, raw_metadata

    saved = metadata.get("fingerprint", None)
    if not isinstance(saved, dict):
        raise ValueError(f"{path.name}: the fingerprint is missing or malformed.")
    for field in FINGERPRINT_FIELDS:
        if saved.get(field) != fingerprint[field]:
            raise ValueError(
                f"{path.name}: cannot load, {field} is {saved.get(field)!r} in "
                f"the file but {fingerprint[field]!r} in the server."
            )

    # Rebuild and check every entry before the live cache changes, so that a
    # bad file cannot leave the cache half full.
    restored = []
    for i, (states, tokens) in enumerate(zip(data["cache"], data["tokens"])):
        tokens = tokens.tolist()
        if len(tokens) != int(metadata["counts"][i]):
            raise ValueError(
                f"{path.name}: entry {i} has {len(tokens)} tokens but the file "
                f"records {metadata['counts'][i]}."
            )
        classes = metadata["classes"][i]
        metas = metadata["meta"][i]
        if not len(classes) == len(metas) == len(states):
            raise ValueError(
                f"{path.name}: entry {i} has {len(states)} layers of data, "
                f"{len(classes)} classes and {len(metas)} metadata."
            )
        caches = []
        for class_name, state, meta_state in zip(classes, states, metas):
            cls = getattr(cache_module, class_name, None)
            if not isinstance(cls, type) or not issubclass(
                cls, cache_module._BaseCache
            ):
                raise ValueError(f"{path.name}: unknown cache class {class_name!r}.")
            caches.append(cls.from_state(state, meta_state))
        # A cache filed under a key that it does not cover reports more cached
        # tokens than it holds, which corrupts the generation.
        for c in caches:
            offset = getattr(c, "offset", None)
            if (
                offset is not None
                and abs(int(offset) - len(tokens)) > _OFFSET_TOLERANCE
            ):
                raise ValueError(
                    f"{path.name}: entry {i} holds {offset} tokens but its key "
                    f"has {len(tokens)}."
                )
        restored.append((tokens, caches, metadata["types"][i]))

    for tokens, caches, cache_type in restored:
        prompt_cache.insert_cache(model_key, tokens, caches, cache_type=cache_type)
    mx.clear_cache()

    return {
        "entries": len(restored),
        "tokens": [len(t) for t, _, _ in restored],
        "types": [t for _, _, t in restored],
        "path": str(path),
    }
