# Copyright © 2026 Apple Inc.

import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm import cache_store
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    LRUPromptCache,
    QuantizedKVCache,
    can_trim_prompt_cache,
)

# The layout of a hybrid model: recurrent layers that hold a fixed state and
# full attention layers that grow with the prompt.
LAYERS = "AAKAAK"
LATENT, ROPE, HEADS, DIM = 128, 64, 2, 4
MODEL_KEY = ("test-model", None, None)
PROMPT = list(range(9000, 9020))


class StubModel:
    def make_cache(self):
        return [ArraysCache(size=2) if k == "A" else KVCache() for k in LAYERS]


def make_layer_caches(n_tokens, bits=4):
    caches = []
    for kind in LAYERS:
        if kind == "A":
            c = ArraysCache(size=2)
            c[0] = mx.random.normal((1, 3, HEADS * DIM)).astype(mx.bfloat16)
            c[1] = mx.random.normal((1, HEADS, DIM, DIM)).astype(mx.float32)
        else:
            c = QuantizedKVCache(group_size=64, bits=bits) if bits else KVCache()
            c.update_and_fetch(
                mx.random.normal((1, 1, n_tokens, LATENT)).astype(mx.bfloat16),
                mx.random.normal((1, 1, n_tokens, ROPE)).astype(mx.bfloat16),
            )
        caches.append(c)
    return caches


def make_prompt_cache_with(n_entries=2, bits=4):
    prompt_cache = LRUPromptCache(max_size=32)
    prompt_cache.insert_cache(
        MODEL_KEY, PROMPT[:8], make_layer_caches(8, bits), cache_type="system"
    )
    if n_entries > 1:
        prompt_cache.insert_cache(
            MODEL_KEY, PROMPT[:16], make_layer_caches(16, bits), cache_type="user"
        )
    return prompt_cache


class TestCacheName(unittest.TestCase):
    def test_rejects_unsafe_names(self):
        for name in ("", "..", "../etc/passwd", "a/b", ".hidden", "x" * 200, "a b"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    cache_store.check_name(name)

    def test_accepts_plain_names(self):
        self.assertEqual(cache_store.check_name("k3-12k_v1.a"), "k3-12k_v1.a")

    def test_path_is_per_rank(self):
        path = cache_store.cache_path("demo", 3, root="/tmp/store")
        self.assertEqual(path.name, "demo-rank3.safetensors")
        self.assertEqual(str(path.parent), "/tmp/store")

    def test_dir_from_environment(self):
        old = os.environ.get("MLX_LM_PROMPT_CACHE_DIR")
        os.environ["MLX_LM_PROMPT_CACHE_DIR"] = "/tmp/from-env"
        try:
            self.assertEqual(str(cache_store.cache_dir()), "/tmp/from-env")
            self.assertEqual(str(cache_store.cache_dir("/tmp/other")), "/tmp/other")
        finally:
            if old is None:
                del os.environ["MLX_LM_PROMPT_CACHE_DIR"]
            else:
                os.environ["MLX_LM_PROMPT_CACHE_DIR"] = old


class TestFingerprint(unittest.TestCase):
    def test_quantized_kv_is_the_same_kind_as_kv(self):
        self.assertEqual(
            cache_store.layer_kinds([QuantizedKVCache(), ArraysCache(size=2)]),
            cache_store.layer_kinds([KVCache(), ArraysCache(size=2)]),
        )

    def test_fields(self):
        fingerprint = cache_store.make_fingerprint(StubModel(), MODEL_KEY, 2, 4)
        self.assertEqual(sorted(fingerprint), sorted(cache_store.FINGERPRINT_FIELDS))
        self.assertEqual(fingerprint["rank"], "2")
        self.assertEqual(fingerprint["world"], "4")
        self.assertEqual(
            fingerprint["layers"],
            "ArraysCache,ArraysCache,KVCache,ArraysCache,ArraysCache,KVCache",
        )


class TestCacheStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        self.fingerprint = cache_store.make_fingerprint(StubModel(), MODEL_KEY, 0, 1)

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name="corpus", rank=0):
        return cache_store.cache_path(name, rank, self.root)

    def save(self, prompt_cache, name="corpus", **kwargs):
        return cache_store.save(
            prompt_cache, MODEL_KEY, self.fingerprint, self.path(name), **kwargs
        )

    def load(self, prompt_cache, name="corpus", fingerprint=None):
        return cache_store.load(
            prompt_cache,
            MODEL_KEY,
            fingerprint or self.fingerprint,
            self.path(name),
        )

    def test_save_reports_the_entries(self):
        report = self.save(make_prompt_cache_with())
        self.assertEqual(report["entries"], 2)
        self.assertEqual(sorted(report["tokens"]), [8, 16])
        self.assertEqual(sorted(report["types"]), ["system", "user"])
        self.assertTrue(Path(report["path"]).is_file())
        self.assertGreater(report["bytes"], 0)
        self.assertEqual(list(Path(self.root).glob("*.tmp*")), [])

    def test_round_trip_restores_a_usable_cache(self):
        self.save(make_prompt_cache_with())
        restored = LRUPromptCache(max_size=32)
        report = self.load(restored)
        self.assertEqual(report["entries"], 2)
        self.assertEqual(len(restored), 2)

        cache, rest = restored.fetch_nearest_cache(MODEL_KEY, PROMPT)
        self.assertIsNotNone(cache)
        self.assertEqual(len(PROMPT) - len(rest), 16)
        self.assertEqual(rest, PROMPT[16:])
        # A cache of recurrent layers cannot be trimmed, so only a full prefix
        # can serve a hit.
        self.assertFalse(can_trim_prompt_cache(cache))

        kv = [c for c in cache if isinstance(c, QuantizedKVCache)]
        self.assertEqual([c.offset for c in kv], [16, 16])
        keys, values = kv[0].update_and_fetch(
            mx.random.normal((1, 1, 1, LATENT)).astype(mx.bfloat16),
            mx.random.normal((1, 1, 1, ROPE)).astype(mx.bfloat16),
        )
        mx.eval(keys, values)
        self.assertEqual(kv[0].offset, 17)
        self.assertEqual(
            mx.dequantize(*keys, group_size=kv[0].group_size, bits=kv[0].bits).shape[
                -2
            ],
            17,
        )

    def test_round_trip_keeps_the_arrays(self):
        prompt_cache = make_prompt_cache_with(n_entries=1)
        _, _, entry = next(prompt_cache.items())
        saved = entry.prompt_cache
        self.save(prompt_cache)
        restored = LRUPromptCache(max_size=32)
        self.load(restored)
        _, _, entry = next(restored.items())
        for before, after in zip(saved, entry.prompt_cache):
            self.assertEqual(type(before), type(after))
            if isinstance(before, ArraysCache):
                self.assertIsNone(after.left_padding)
                self.assertIsNone(after.lengths)
                for a, b in zip(before.cache, after.cache):
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertTrue(mx.array_equal(a, b))
                self.assertEqual(before.nbytes, after.nbytes)
            else:
                # A KV cache grows in steps of 256 tokens but it saves only
                # the tokens that it holds, so the file is smaller.
                self.assertEqual(before.offset, after.offset)
                self.assertLess(after.nbytes, before.nbytes)

    def test_load_of_a_quantized_cache_into_an_unquantized_server(self):
        # The layer kinds match, so a cache that quantized itself while the
        # server ran still loads.
        self.save(make_prompt_cache_with(bits=4))
        restored = LRUPromptCache(max_size=32)
        self.load(restored)
        self.assertEqual(len(restored), 2)

    def test_load_refuses_another_model(self):
        self.save(make_prompt_cache_with())
        other = dict(self.fingerprint, model=repr(("other-model", None, None)))
        restored = LRUPromptCache(max_size=32)
        with self.assertRaises(ValueError):
            self.load(restored, fingerprint=other)
        self.assertEqual(len(restored), 0)

    def test_load_refuses_another_layout_or_rank(self):
        self.save(make_prompt_cache_with())
        for field, value in (("layers", "KVCache"), ("rank", "1"), ("world", "8")):
            with self.subTest(field=field):
                restored = LRUPromptCache(max_size=32)
                with self.assertRaises(ValueError):
                    self.load(
                        restored, fingerprint=dict(self.fingerprint, **{field: value})
                    )
                self.assertEqual(len(restored), 0)

    def test_load_refuses_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.load(LRUPromptCache(max_size=32), name="not-there")

    def test_load_refuses_a_cache_that_does_not_cover_its_key(self):
        prompt_cache = LRUPromptCache(max_size=32)
        # A key of 50 tokens but a cache of only 16 tokens.
        prompt_cache.insert_cache(
            MODEL_KEY, list(range(50)), make_layer_caches(16), cache_type="user"
        )
        self.save(prompt_cache, name="bad")
        restored = LRUPromptCache(max_size=32)
        with self.assertRaises(ValueError):
            self.load(restored, name="bad")
        self.assertEqual(len(restored), 0)

    def test_save_skips_other_models_and_empty_entries(self):
        prompt_cache = make_prompt_cache_with()
        prompt_cache.insert_cache(
            ("another-model", None, None), [1, 2, 3], make_layer_caches(3)
        )
        prompt_cache.insert_cache(MODEL_KEY, [4, 5, 6], StubModel().make_cache())
        report = self.save(prompt_cache)
        self.assertEqual(report["entries"], 2)
        self.assertEqual(report["skipped"], 2)

    def test_save_filters(self):
        report = self.save(make_prompt_cache_with(), name="a", types=["system"])
        self.assertEqual(report["types"], ["system"])
        self.assertEqual(report["skipped"], 1)

        report = self.save(make_prompt_cache_with(), name="b", min_tokens=12)
        self.assertEqual(report["tokens"], [16])
        self.assertEqual(report["skipped"], 1)

        with self.assertRaises(ValueError):
            self.save(make_prompt_cache_with(), name="c", min_tokens=10000)

    def test_save_of_an_empty_cache_is_refused(self):
        with self.assertRaises(ValueError):
            self.save(LRUPromptCache(max_size=32))
        self.assertFalse(self.path().exists())

    def test_save_replaces_a_file_and_leaves_no_partial_file(self):
        self.save(make_prompt_cache_with())
        self.save(make_prompt_cache_with(n_entries=1))
        self.assertEqual(list(Path(self.root).glob("*.tmp*")), [])
        restored = LRUPromptCache(max_size=32)
        self.assertEqual(self.load(restored)["entries"], 1)

    def test_saved_names_and_listing(self):
        self.save(make_prompt_cache_with(), name="one")
        self.save(make_prompt_cache_with(), name="two")
        cache_store.save(
            make_prompt_cache_with(),
            MODEL_KEY,
            self.fingerprint,
            cache_store.cache_path("three", 1, self.root),
        )
        self.assertEqual(cache_store.saved_names(0, self.root), ["one", "two"])
        listed = cache_store.listing(make_prompt_cache_with(), 0, self.root)
        self.assertEqual(listed["saved"], ["one", "two"])
        self.assertEqual(listed["rank"], 0)
        self.assertEqual(listed["in_memory"]["sequences"], 2)
        self.assertIn("system", listed["in_memory"]["by_type"])


if __name__ == "__main__":
    unittest.main()
