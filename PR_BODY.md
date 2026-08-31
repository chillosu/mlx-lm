# Save and load the server prompt cache on disk

## Problem

`mlx_lm.server` keeps prefilled KV caches in an `LRUPromptCache` and reuses
them across requests, which is what makes a second question against the same
corpus fast. That cache exists only in memory, so it dies with the process.

On a 4-node tensor-parallel Kimi-K3 setup the prefill of a 201K token corpus
costs **2,802 s** — 47 minutes of wall clock that a restart, an OOM, a crash
or a config change throws away. A 949K token corpus would cost hours. There is
no way to get the cache out of the process, and `save_prompt_cache` /
`load_prompt_cache` only handle one flat cache list, not the server's trie of
entries, and know nothing about ranks.

## What this adds

Four endpoints on the server:

```
POST /admin/cache/save   {"name": ..., "types": [...], "min_tokens": ...}
POST /admin/cache/load   {"name": ...}
POST /admin/cache/clear  {}
GET  /admin/cache
```

plus `--prompt-cache-dir` (falls back to `$MLX_LM_PROMPT_CACHE_DIR`, then
`~/.cache/mlx-lm/prompt-cache`), a `LRUPromptCache.items()` iterator, and a new
`mlx_lm/cache_store.py` that holds the archive format and its guards.

## Mechanism

Only rank 0 runs the HTTP server, and all ranks meet exactly once per turn of
the generation loop, in `_share_request`. So a save or a load is a
`PromptCacheRequest` put on the ordinary request queue: rank 0 enqueues it,
`_share_request` broadcasts it, and every rank runs the operation at the same
point of the loop and writes its own shard to its own local disk
(`<name>-rank<N>.safetensors`) — no cross-rank file shipping. Each rank
reduces a success flag through an `all_sum` barrier so all ranks take the same
branch; the *number* of barriers depends only on values that are identical
across ranks (the shared request, `len(batch_results)`), never on one rank's
outcome, which is the property that keeps the collectives in step. A listing
mutates nothing and reads no other rank, so rank 0 answers it directly without
touching the loop.

Guards: fingerprint match (model key, rank, world size, per-layer cache kinds),
atomic staged write plus `os.replace`, a presence barrier before a load mutates
anything, rollback on every rank if any rank fails, per-entry validation
(token count, and every KV layer's offset must agree with its trie key) before
a single `insert_cache`, name sanitising against path traversal, a free-space
check, and `409` while any generation is in flight.

## Measured

4× Mac Studio M3 Ultra, tensor parallel over 4 ranks, Kimi K3 MXFP4, mlx
0.32.2, 4-bit KV cache. "Third question" means a question never asked before
the restart, so it can only be served by the restored cache.

| corpus | cold prefill | warm, in memory | save | restart | load | third question |
|---|---|---|---|---|---|---|
| 12,664 tok | 338 s | 100 s | — | `pkill -9` on **all four** ranks | from disk | `cached_tokens` 12,644 of 12,664, prefill `20/20 tokens`, **50 s** wall |
| 201,186 tok | **2,802 s** | 93 s (`cached_tokens` 201,151) | 10.47 GB/rank in **2 s** | full 4-rank restart | 202 s | `cached_tokens` 201,151, **50 s** wall, correct answer |

Storage and throughput, per rank, 4-bit KV:

- **55.9 KB per token per rank.** 201K tokens → 10.47 GB per rank, ~42 GB for
  the fleet.
- **Save ~5 GB/s** to local NVMe (10.47 GB in 2 s).
- The archive slices each KV cache to its offset, so it drops the 256-token
  step padding and the file is smaller than the live cache.

So a 47-minute prefill becomes a 2-second save and a 50-second answer.

## Testing

Two new test files, 44 tests, no model and no distributed hardware needed:

- `tests/test_cache_store.py` (19 tests) — archive round trip for
  `ArraysCache` / `KVCache` / `QuantizedKVCache`, a reloaded quantized cache
  taking a decode step, fingerprint refusals, filters, atomic overwrite,
  offset/key mismatch, listing.
- `tests/test_server_distributed.py` (25 tests) — a threaded fake-distributed
  harness. Four threads stand in for four ranks, with `threading.Barrier`
  backed `all_sum`, so a path that is *not* symmetric across ranks surfaces as
  a broken barrier instead of a hang. It covers: save and load across four
  ranks, one rank's file missing, one rank's file carrying a foreign
  fingerprint, one rank unable to write, a bad name, and the in-flight guard —
  asserting in every case that all ranks ran the same number of barriers and
  ended in the same state. A separate case drives the *real* `_generate` loop
  on four threads and pushes requests through `_share_request` exactly as the
  HTTP handler does, and an HTTP case serves a real `APIHandler` over a socket.

```
python -m unittest tests.test_cache_store tests.test_server_distributed
# Ran 44 tests ... OK

python -m unittest tests.test_cache_store tests.test_server_distributed \
                   tests.test_prompt_cache tests.test_server
# Ran 92 tests ... OK
```

`uvx pre-commit run --all` is clean.

Beyond the unit tests, the mechanism ran in production on the 4-node fleet
described above: repeated save/load cycles, a `pkill -9` on all four ranks, and
a full fleet restart, each followed by a question that had never been asked.

## Limitations

- **This is checkpoint and restore, not paging.** In-memory serving is
  unchanged. Nothing spills to disk under memory pressure and nothing is
  faulted back in on a miss; a human (or a script) decides when to save and
  when to load.
- **The corpus has to be in the system message** to get reuse across different
  questions on models whose caches cannot be trimmed. Kimi K3 mixes recurrent
  layers (`ArraysCache`, `is_trimmable()` is False) with full-attention layers,
  so `can_trim_prompt_cache` is False and `fetch_nearest_cache` only serves an
  exact match or a full prefix. With the corpus in a user message the stored
  key is `corpus + question A`, which is not a prefix of `corpus + question B`,
  and an entry sharing 12,600 of 12,644 tokens is worth nothing. This is a
  property of the existing cache, not of this change, but it decides whether
  the restored cache is useful.
- **The 202 s load is not I/O.** The same fleet takes 170 s for an admin op
  that does no work at all and hits the 120 s timeout on a `clear`, while a
  10 GB save takes 2 s. The time goes into a single idle collective in
  `_share_request` that blocks for minutes on an otherwise idle fleet. That is
  upstream code and an existing problem — it stalls completions the same way —
  and it is only visible here because an admin op is often the first request to
  arrive on a genuinely idle fleet. The second commit reduces exposure to it
  and reports it; it does not fix it, and the root cause is being reported
  separately.
- **The fingerprint is structural, not exhaustive.** It pins the model key, the
  rank, the world size and the per-layer cache kinds, with a quantized KV cache
  counted as the same kind as an unquantized one — a layer can quantize its own
  cache while the server runs, and a restored quantized cache carries its own
  group size and bits, so it keeps working. There is no check that the four
  shards came from the *same* prefill: four shards from four different but
  identically configured prefills would load without complaint.
- **A save is a snapshot, not a log.** Re-saving rewrites the file, and a load
  appends to whatever is already in memory — use `clear` first for a clean
  state.
- **Peak RSS during a save is about the file size**, because slicing each cache
  to its offset materialises a copy. `mx.clear_cache()` runs right after, but a
  10 GB save needs 10 GB of headroom on each node.
- **Only tensor parallelism was measured.** The mechanism does not depend on
  the parallelism mode — it rides the request path — but `--pipeline` was not
  tested.

## Second commit — the idle poll

`MLX_LM_IDLE_POLL_S` (default 1.0 s, was a hard-coded 0.1 s) is independent of
the feature and can be dropped without affecting it. Rationale: every turn of
the idle generation loop ends in a collective that joins all ranks, so a 0.1 s
queue poll makes an idle fleet meet ten times a second to do nothing, and any
request that arrives while one of those collectives blocks waits it out.
`Queue.get` wakes on `put`, so a longer poll adds **no** pickup latency (there
is a test for exactly that); it only makes the server take up to the poll
interval to stop. `MLX_LM_IDLE_POLL_S=0.1` restores today's behaviour.

## Notes on reproduction

The fleet these numbers come from also runs a small local patch that adds
`--kv-bits` / `--kv-group-size` / `--quantized-kv-start` to `mlx_lm.server`,
which upstream does not have; that is how the KV cache was 4-bit. Nothing in
this PR depends on those flags — the fingerprint deliberately does not read
them — but the byte-per-token figures above are 4-bit figures and would be
roughly 4× larger with an unquantized KV cache.
