# Copyright © 2026 Apple Inc.

"""Tests for the prompt cache endpoints of the server.

No model is loaded. The caches hold small arrays with the shape of a hybrid
model, and a group of threads stands in for the ranks of a distributed server.
"""

import json
import logging
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import mlx.core as mx

from mlx_lm import cache_store, server
from mlx_lm.models.cache import ArraysCache, KVCache, LRUPromptCache, QuantizedKVCache
from mlx_lm.server import (
    _SLOW_SYNC_WARN_S,
    APIHandler,
    PromptCacheRequest,
    ResponseGenerator,
)

LAYERS = "AAKAAK"
LATENT, ROPE, HEADS, DIM = 128, 64, 2, 4
MODEL_KEY = ("test-model", None, None)
PROMPT = list(range(9000, 9020))
WORLD = 4

_thread_local = threading.local()


class StubModel:
    def make_cache(self):
        return [ArraysCache(size=2) if k == "A" else KVCache() for k in LAYERS]


class StubProvider:
    def __init__(self, prompt_cache_dir=None):
        self.model = StubModel()
        self.model_key = MODEL_KEY
        self.cli_args = type(
            "Args",
            (object,),
            {
                "allowed_origins": ["*"],
                "prompt_cache_dir": prompt_cache_dir,
                "prompt_cache_bytes": None,
                "model": None,
            },
        )

    def load_default(self):
        return None


def make_layer_caches(n_tokens):
    caches = []
    for kind in LAYERS:
        if kind == "A":
            c = ArraysCache(size=2)
            c[0] = mx.random.normal((1, 3, HEADS * DIM)).astype(mx.bfloat16)
            c[1] = mx.random.normal((1, HEADS, DIM, DIM)).astype(mx.float32)
        else:
            c = QuantizedKVCache(group_size=64, bits=4)
            c.update_and_fetch(
                mx.random.normal((1, 1, n_tokens, LATENT)).astype(mx.bfloat16),
                mx.random.normal((1, 1, n_tokens, ROPE)).astype(mx.bfloat16),
            )
        caches.append(c)
    return caches


def populate(prompt_cache):
    """Insert the two segments that a chat prompt leaves in the cache."""
    prompt_cache.insert_cache(
        MODEL_KEY, PROMPT[:8], make_layer_caches(8), cache_type="system"
    )
    prompt_cache.insert_cache(
        MODEL_KEY, PROMPT[:16], make_layer_caches(16), cache_type="user"
    )


def quiet_errors(test):
    """The tests below check errors that the server reports, so do not log."""
    logging.disable(logging.ERROR)
    test.addCleanup(logging.disable, logging.NOTSET)


def make_generator(prompt_cache, provider, rank=0, distributed=False):
    """A ResponseGenerator without a generation thread and without a model."""
    generator = ResponseGenerator.__new__(ResponseGenerator)
    generator.prompt_cache = prompt_cache
    generator.model_provider = provider
    generator.requests = queue.Queue()
    generator._state_machine_cache = {}
    generator._is_distributed = distributed
    generator._rank = rank
    generator._stop = False
    return generator


def serve(generator, op, name="", in_flight=0, **kwargs):
    """Run one operation and return what the handler answers."""
    rqueue = queue.Queue()
    generator._serve_prompt_cache(
        rqueue, PromptCacheRequest(op=op, name=name, **kwargs), in_flight=in_flight
    )
    result = rqueue.get()
    assert rqueue.get() is None
    return result


class FakeGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class FakeDistributed:
    """A stand in for mx.distributed that sums over threads."""

    def __init__(self, world):
        self.world = world
        self.enter = threading.Barrier(world, timeout=10)
        self.leave = threading.Barrier(world, timeout=10)
        self.slots = [None] * world
        self.calls = [0] * world

    def init(self, *args, **kwargs):
        return FakeGroup(self.world, getattr(_thread_local, "rank", 0))

    def all_sum(self, x, **kwargs):
        rank = getattr(_thread_local, "rank", 0)
        self.calls[rank] += 1
        value = x if isinstance(x, mx.array) else mx.array(x)
        mx.eval(value)
        self.slots[rank] = value
        self.enter.wait()
        total = self.slots[0]
        for i in range(1, self.world):
            total = total + self.slots[i]
        mx.eval(total)
        self.leave.wait()
        return total


class TestPromptCacheAdmin(unittest.TestCase):
    """One rank, so the barriers are local and every branch is reachable."""

    def setUp(self):
        quiet_errors(self)
        self.dir = TemporaryDirectory()
        self.provider = StubProvider(self.dir.name)
        self.prompt_cache = LRUPromptCache(max_size=32)
        populate(self.prompt_cache)
        self.generator = make_generator(self.prompt_cache, self.provider)

    def tearDown(self):
        self.dir.cleanup()

    def test_save_then_load_into_a_fresh_cache(self):
        report = serve(self.generator, "save", "corpus")
        self.assertEqual(report["entries"], 2)

        fresh = make_generator(LRUPromptCache(max_size=32), StubProvider(self.dir.name))
        report = serve(fresh, "load", "corpus")
        self.assertEqual(report["entries"], 2)
        self.assertEqual(len(fresh.prompt_cache), 2)

        cache, rest = fresh.prompt_cache.fetch_nearest_cache(MODEL_KEY, PROMPT)
        self.assertIsNotNone(cache)
        self.assertEqual(len(PROMPT) - len(rest), 16)

    def test_load_of_a_name_that_was_never_saved(self):
        result = serve(self.generator, "load", "not-there")
        self.assertIsInstance(result, FileNotFoundError)

    def test_unsafe_name(self):
        for op in ("save", "load"):
            result = serve(self.generator, op, "../etc/passwd")
            self.assertIsInstance(result, ValueError)
        self.assertEqual(list(Path(self.dir.name).iterdir()), [])

    def test_operations_are_refused_while_a_generation_is_in_flight(self):
        for op in ("save", "load"):
            result = serve(self.generator, op, "corpus", in_flight=2)
            self.assertIsInstance(result, RuntimeError)
        self.assertFalse(list(Path(self.dir.name).glob("*.safetensors")))

    def test_save_filters(self):
        report = serve(self.generator, "save", "only-system", types=["system"])
        self.assertEqual(report["types"], ["system"])
        self.assertIsInstance(
            serve(self.generator, "save", "unknown-type", types=["nope"]), ValueError
        )
        self.assertIsInstance(
            serve(self.generator, "save", "too-long", min_tokens=10000), ValueError
        )

    def test_clear_empties_the_cache(self):
        self.assertEqual(serve(self.generator, "clear")["cleared"], 2)
        self.assertEqual(len(self.prompt_cache), 0)
        self.assertIsInstance(serve(self.generator, "save", "empty"), ValueError)

    def test_list_reports_the_saved_and_the_live_caches(self):
        serve(self.generator, "save", "corpus")
        result = serve(self.generator, "list")
        self.assertEqual(result["saved"], ["corpus"])
        self.assertEqual(result["in_memory"]["sequences"], 2)

    def test_unknown_operation(self):
        self.assertIsInstance(serve(self.generator, "bogus"), ValueError)

    def test_the_request_can_be_pickled(self):
        # The request travels to the other ranks as a pickle.
        import pickle

        request = pickle.loads(pickle.dumps(PromptCacheRequest(op="save", name="x")))
        self.assertEqual((request.op, request.name), ("save", "x"))


class TestPromptCacheFleet(unittest.TestCase):
    """Every rank must leave an operation on the same branch.

    A rank that runs another number of barriers puts the collectives of the
    generation loop out of step and the server stops to answer. Here a
    threading.Barrier stands in for the collective, so an operation that is not
    the same on every rank breaks the barrier instead of hanging the test.
    """

    def setUp(self):
        quiet_errors(self)
        self.dir = TemporaryDirectory()
        self.roots = [str(Path(self.dir.name) / f"rank{r}") for r in range(WORLD)]
        for root in self.roots:
            Path(root).mkdir()
        self.fake = FakeDistributed(WORLD)
        self.patcher = mock.patch.object(mx, "distributed", self.fake)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.dir.cleanup()

    def path(self, name, rank):
        return cache_store.cache_path(name, rank, self.roots[rank])

    def run_fleet(self, op, name="", populated=True, in_flight=0, roots=None):
        roots = roots or self.roots
        results = [None] * WORLD
        caches = [None] * WORLD
        failures = [None] * WORLD
        counts = [0] * WORLD
        barrier = threading.Barrier(WORLD, timeout=10)
        flags = [None] * WORLD

        def rank_barrier(rank):
            def _barrier(ok):
                flags[rank] = bool(ok)
                counts[rank] += 1
                barrier.wait()
                decision = all(flags)
                barrier.wait()
                return decision

            return _barrier

        def run(rank):
            _thread_local.rank = rank
            prompt_cache = LRUPromptCache(max_size=32)
            if populated:
                populate(prompt_cache)
            caches[rank] = prompt_cache
            generator = make_generator(
                prompt_cache, StubProvider(roots[rank]), rank=rank, distributed=True
            )
            generator._prompt_cache_barrier = rank_barrier(rank)
            try:
                results[rank] = serve(generator, op, name, in_flight=in_flight)
            except BaseException as e:  # a broken barrier, not a hang
                failures[rank] = e

        threads = [threading.Thread(target=run, args=(r,)) for r in range(WORLD)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertFalse([t for t in threads if t.is_alive()], "a rank did not finish")
        self.assertEqual(failures, [None] * WORLD)
        self.assertEqual(len(set(counts)), 1, f"the ranks ran {counts} barriers")
        return results, caches, counts

    def test_save_and_load_on_every_rank(self):
        results, _, counts = self.run_fleet("save", "corpus")
        self.assertTrue(all(isinstance(r, dict) for r in results))
        self.assertEqual(counts[0], 1)
        self.assertTrue(all(self.path("corpus", r).is_file() for r in range(WORLD)))

        results, caches, counts = self.run_fleet("load", "corpus", populated=False)
        self.assertTrue(all(isinstance(r, dict) for r in results))
        # One barrier for the files and one for the result.
        self.assertEqual(counts[0], 2)
        self.assertTrue(all(len(c) == 2 for c in caches))
        hits = [
            len(PROMPT) - len(c.fetch_nearest_cache(MODEL_KEY, PROMPT)[1])
            for c in caches
        ]
        self.assertEqual(hits, [16] * WORLD)

    def test_a_missing_file_stops_the_load_on_every_rank(self):
        self.run_fleet("save", "corpus")
        self.path("corpus", 2).unlink()
        results, caches, counts = self.run_fleet("load", "corpus", populated=False)
        self.assertTrue(all(isinstance(r, FileNotFoundError) for r in results))
        self.assertEqual(counts[0], 2)
        self.assertTrue(all(len(c) == 0 for c in caches))

    def test_a_bad_file_on_one_rank_puts_every_rank_back(self):
        self.run_fleet("save", "corpus")
        # Give rank 2 a file from another model.
        prompt_cache = LRUPromptCache(max_size=32)
        populate(prompt_cache)
        cache_store.save(
            prompt_cache,
            MODEL_KEY,
            dict(
                cache_store.make_fingerprint(StubModel(), MODEL_KEY, 2, WORLD),
                model=repr(("other-model", None, None)),
            ),
            self.path("corpus", 2),
        )
        results, caches, _ = self.run_fleet("load", "corpus", populated=False)
        self.assertTrue(all(isinstance(r, Exception) for r in results))
        self.assertIn("model is", str(results[2]))
        self.assertTrue(all(len(c) == 0 for c in caches))

    def test_a_rank_that_cannot_write_removes_the_files_of_the_others(self):
        roots = list(self.roots)
        roots[2] = "/dev/null/nope"
        results, _, _ = self.run_fleet("save", "corpus", roots=roots)
        self.assertTrue(all(isinstance(r, Exception) for r in results))
        self.assertFalse(any(self.path("corpus", r).exists() for r in (0, 1, 3)))

    def test_a_bad_name_stops_before_the_file_barrier(self):
        results, _, counts = self.run_fleet("load", "../etc/passwd")
        self.assertTrue(all(isinstance(r, ValueError) for r in results))
        self.assertEqual(counts[0], 1)

    def test_the_in_flight_guard_is_the_same_on_every_rank(self):
        results, _, counts = self.run_fleet("save", "corpus", in_flight=2)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
        self.assertEqual(counts[0], 1)
        self.assertFalse(any(self.path("corpus", r).exists() for r in range(WORLD)))


class TestPromptCacheGenerationLoop(unittest.TestCase):
    """Push a request through the real generation loop of four ranks.

    The request travels the same path as a completion: rank 0 puts it in the
    queue and _share_request sends it to every rank.
    """

    def setUp(self):
        self.dir = TemporaryDirectory()
        # Write the file of every rank here. The generation thread of a rank
        # must build the arrays that it saves, because mlx keeps a stream per
        # thread.
        prompt_cache = LRUPromptCache(max_size=32)
        populate(prompt_cache)
        for rank in range(WORLD):
            cache_store.save(
                prompt_cache,
                MODEL_KEY,
                cache_store.make_fingerprint(StubModel(), MODEL_KEY, rank, WORLD),
                cache_store.cache_path("corpus", rank, self.dir.name),
            )

        self.fake = FakeDistributed(WORLD)
        self.patcher = mock.patch.object(mx, "distributed", self.fake)
        self.patcher.start()
        self.generators = []
        self.threads = []
        for rank in range(WORLD):
            generator = make_generator(
                LRUPromptCache(max_size=32),
                StubProvider(self.dir.name),
                rank=rank,
                distributed=True,
            )
            generator._time_budget = None
            self.generators.append(generator)

        def run(rank):
            _thread_local.rank = rank
            try:
                self.generators[rank]._generate()
            except BaseException:
                pass

        for rank in range(WORLD):
            thread = threading.Thread(target=run, args=(rank,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def tearDown(self):
        for generator in self.generators:
            generator._stop = True
        for thread in self.threads:
            thread.join(timeout=30)
        self.patcher.stop()
        self.dir.cleanup()

    def send(self, op, name="", timeout=20):
        rqueue = queue.Queue()
        self.generators[0].requests.put(
            (rqueue, PromptCacheRequest(op=op, name=name), None)
        )
        outcome = rqueue.get(timeout=timeout)
        self.assertIsNone(rqueue.get(timeout=5))
        return outcome

    def test_the_idle_loop_keeps_the_ranks_together(self):
        before = list(self.fake.calls)

        def ticking():
            return all(a > b for a, b in zip(self.fake.calls, before))

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not ticking():
            time.sleep(0.1)
        self.assertTrue(ticking(), "the ranks do not exchange collectives")
        self.assertTrue(all(t.is_alive() for t in self.threads))

    def test_a_request_travels_the_loop_and_the_ranks_stay_healthy(self):
        self.assertEqual(self.send("list")["saved"], ["corpus"])

        result = self.send("load", "corpus")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["entries"], 2)
        self.assertTrue(all(len(g.prompt_cache) == 2 for g in self.generators))

        result = self.send("save", "again")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["entries"], 2)

        self.assertEqual(self.send("clear")["cleared"], 2)
        self.assertTrue(all(len(g.prompt_cache) == 0 for g in self.generators))
        self.assertTrue(all(t.is_alive() for t in self.threads))

    def test_a_long_idle_poll_does_not_delay_a_request(self):
        # A queue wakes on a put, so the loop picks a request up at once even
        # when the idle poll is a full second.
        latencies = []
        for _ in range(3):
            start = time.monotonic()
            self.send("list")
            latencies.append(time.monotonic() - start)
            time.sleep(0.2)
        self.assertLess(max(latencies), server._IDLE_POLL_S / 2)


class TestSlowRankSync(unittest.TestCase):
    def test_a_slow_share_request_is_reported(self):
        generator = make_generator(LRUPromptCache(max_size=4), StubProvider())
        share_request = generator._share_request
        generator._share_request = lambda r: (
            time.sleep(_SLOW_SYNC_WARN_S + 0.2),
            share_request(r),
        )[1]
        with self.assertLogs(level=logging.WARNING) as logs:
            generator._next_request(timeout=0.01)
        message = "\n".join(logs.output)
        self.assertIn("Slow rank sync", message)
        self.assertIn("rank 0", message)
        self.assertIn("had_request=False", message)

    def test_a_normal_share_request_is_not_reported(self):
        generator = make_generator(LRUPromptCache(max_size=4), StubProvider())
        with self.assertNoLogs(level=logging.WARNING):
            generator._next_request(timeout=0.01)


class DeadGenerator:
    """A ResponseGenerator whose loop does not run. Nothing reads `requests`."""

    def __init__(self, root):
        self.prompt_cache = LRUPromptCache(max_size=32)
        populate(self.prompt_cache)
        self.requests = queue.Queue()
        self.cli_args = StubProvider(root).cli_args
        self._rank = 0


class TestPromptCacheHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = TemporaryDirectory()
        cls.generator = DeadGenerator(cls.dir.name)
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: APIHandler(
                cls.generator, *args, system_fingerprint="test", **kwargs
            ),
        )
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.dir.cleanup()

    def test_the_listing_answers_without_the_generation_loop(self):
        # A listing reads only the state of rank 0, so it must not wait for
        # the generation loop.
        for path in ("/admin/cache", "/admin/cache/", "/admin/cache/list"):
            with self.subTest(path=path):
                start = time.monotonic()
                with urllib.request.urlopen(self.url + path, timeout=10) as response:
                    body = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertLess(time.monotonic() - start, 2.0)
                self.assertEqual(body["op"], "list")
                self.assertEqual(body["result"]["rank"], 0)
                self.assertEqual(body["result"]["in_memory"]["sequences"], 2)
        self.assertTrue(self.generator.requests.empty())

    def test_an_unknown_path_is_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(self.url + "/admin/cache/bogus", timeout=5)
        self.assertEqual(raised.exception.code, 404)

    def post(self, op, body, answer):
        """POST an operation and let a thread answer it as a rank would."""

        def reply():
            rqueue, request, _ = self.generator.requests.get(timeout=10)
            self.replied = request
            rqueue.put(answer)
            rqueue.put(None)

        thread = threading.Thread(target=reply)
        thread.start()
        try:
            request = urllib.request.Request(
                self.url + f"/admin/cache/{op}",
                data=json.dumps(body).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(request, timeout=10)
        finally:
            thread.join(timeout=10)

    def test_a_save_goes_to_the_generation_loop(self):
        with self.post("save", {"name": "corpus"}, {"entries": 2}) as response:
            body = json.loads(response.read().decode())
        self.assertEqual(response.status, 200)
        self.assertEqual(self.replied.op, "save")
        self.assertEqual(self.replied.name, "corpus")
        self.assertEqual(body["result"], {"entries": 2})

    def test_an_error_from_a_rank_becomes_a_status_code(self):
        for error, status in (
            (ValueError("bad"), 400),
            (FileNotFoundError("gone"), 404),
            (OSError("full"), 507),
            (RuntimeError("busy"), 409),
        ):
            with self.subTest(status=status):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.post("load", {"name": "corpus"}, error)
                self.assertEqual(raised.exception.code, status)

    def test_a_bad_body_is_a_bad_request(self):
        for body in (b"{not json}", json.dumps({"types": 3}).encode()):
            with self.subTest(body=body):
                request = urllib.request.Request(
                    self.url + "/admin/cache/save",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 400)
        self.assertTrue(self.generator.requests.empty())


if __name__ == "__main__":
    unittest.main()
