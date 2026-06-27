# Real Paged KV Allocator

## Goal

TurboInfer now includes a real paged KV metadata allocator. This is the first step from paged KV simulation toward a vLLM-like memory-management path.

The allocator manages:

- fixed-size KV blocks;
- per-request block tables;
- per-request context lengths;
- logical token to physical `(block_id, offset)` lookup;
- padded decode metadata export for active request batches;
- append-token growth during decode;
- request cleanup and block reuse;
- utilization and fragmentation metrics.

It does not yet replace Hugging Face's `past_key_values` tensors. Instead, it runs alongside the continuous batching server as real allocator metadata. This is intentional: the next step is to connect this metadata to a paged decode attention kernel.

The important runtime contract is:

- `token_slot(request_id, token_index)` maps a logical token position to a physical KV block and offset;
- `decode_metadata(request_ids)` exports `block_table` with shape `[batch_size, max_blocks_per_request]`;
- `decode_metadata(request_ids)` exports `context_lens` with shape `[batch_size]`.

This is the metadata shape a paged decode attention kernel needs. The current implementation still uses Hugging Face's contiguous legacy cache for actual attention computation, but the allocator now has the same request-to-block indirection layer that a real paged KV path will consume.

The allocator benchmark includes two baselines:

- `contiguous_full_reservation`: a coarse full-workload reservation baseline;
- `dynamic_contiguous_reservation`: the same arrival/decode/free policy as the paged allocator, but each live request reserves `max_sequence_tokens` slots.

## Files

- `src/turboinfer/paged_allocator.py`
- `tests/test_paged_allocator.py`
- `benchmarks/bench_paged_allocator.py`
- `src/turboinfer/continuous.py`

## Run Unit Tests

```bash
pytest tests/test_paged_allocator.py
```

## Run Allocator Benchmark

```bash
python benchmarks/bench_paged_allocator.py \
  --num-requests 32 \
  --arrival-interval-steps 4 \
  --short-prompt-tokens 128 \
  --long-prompt-tokens 2048 \
  --short-output-tokens 64 \
  --long-output-tokens 256 \
  --block-size 16 \
  --total-blocks 2048 \
  --max-sequence-tokens 2304
```

## Run With Continuous Server

Start the server:

```bash
python scripts/start_server_background.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 8 \
  --batch-wait-seconds 0.002 \
  --kv-block-size 16 \
  --kv-total-blocks 4096 \
  --preload
```

Then run:

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine continuous \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

Each response's metrics includes:

```json
{
  "paged_kv_allocator": {
    "block_size": 16,
    "total_blocks": 4096,
    "used_blocks": 0,
    "free_blocks": 4096,
    "peak_used_blocks": 40,
    "peak_live_requests": 8,
    "utilization": 1.0
  }
}
```

The final `used_blocks` may be zero because completed requests release their blocks. The important values are the cumulative counters and peak metrics.

## Interview Framing

This allocator is not yet PagedAttention. It is the metadata layer that PagedAttention needs.

Correct explanation:

> I first simulated paged KV cache to understand memory waste. Then I implemented a real block allocator with request block tables and context lengths. It is integrated with the continuous batching server as metadata. The next step is to replace HF legacy `past_key_values` padding with a paged decode attention kernel that consumes these block tables.

If asked whether this already accelerates attention, the honest answer is no. It reduces the project risk for the real acceleration step by making request admission, decode growth, cleanup, block reuse, and kernel metadata explicit and testable before touching the attention kernel.
