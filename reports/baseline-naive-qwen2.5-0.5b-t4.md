# Baseline Result: Naive No KV Cache

## Run Context

- Date: 2026-06-26
- Engine: TurboInfer
- Optimization: `naive_no_kv_cache`
- Model: `Qwen/Qwen2.5-0.5B`
- Device: Colab T4 GPU
- Prompt: `Explain why KV cache improves LLM decoding.`
- Max new tokens: 64

## Output

```text
KV cache improves LLM decoding by providing a fast and efficient way to store and retrieve data from a cache. This is particularly useful in large-scale language models, where the amount of data that needs to be processed and stored is very large. By caching the results of previous computations, the LLM can quickly access the data
```

## Metrics

| Metric | Value |
| --- | ---: |
| Prompt tokens | 10 |
| Output tokens | 64 |
| Total seconds | 2.9883 |
| TTFT seconds | 0.8860 |
| TPOT seconds | 0.0334 |
| Tokens/s | 21.4170 |
| Peak memory MB | 1002.2856 |

## Initial Interpretation

This is the project baseline. It intentionally disables KV Cache and recomputes the full context at every decode step.

The next KV Cache optimization should be compared against this exact workload:

- Same model: `Qwen/Qwen2.5-0.5B`
- Same prompt
- Same `max_new_tokens=64`
- Same device class: Colab T4

Expected direction after enabling KV Cache:

- TPOT should decrease.
- Tokens/s should increase.
- Peak memory may increase because key/value tensors are stored.
- TTFT may not improve much, because prefill still needs to process the prompt.

