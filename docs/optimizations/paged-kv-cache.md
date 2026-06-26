# Optimization: Paged KV Cache

## 本轮目标

解释并模拟 PagedAttention 背后的核心内存管理思想：不要为每个请求预留一整段最大长度的连续 KV cache，而是把 KV cache 切成固定大小的 blocks，让请求按需持有 blocks。

这一版先实现 block manager simulator，不直接实现 PagedAttention kernel。

## 为什么需要它

Dynamic continuous batching 会带来动态 active set：

- 请求到达时间不同
- prompt 长度不同
- output 长度不同
- 请求完成时间不同
- KV cache 需要随请求生命周期分配和释放

如果每个请求都按最大序列长度预留连续 KV cache，短请求会浪费大量 token slots。Paged KV Cache 用 block table 把逻辑 token 位置映射到物理 KV blocks，从而减少浪费并支持更灵活的分配释放。

## 实现内容

- `PagedKVCacheManager`
- 固定大小 KV blocks
- request 到 block list 的映射
- append token 时按需申请新 block
- request 完成时释放 blocks
- contiguous allocation vs paged allocation 对比

## 运行命令

```bash
python benchmarks/simulate_paged_kv.py \
  --num-requests 16 \
  --short-prompt-tokens 128 \
  --long-prompt-tokens 2048 \
  --output-tokens 128 \
  --block-size 16 \
  --max-sequence-tokens 2176
```

## 观察指标

- `allocated_token_slots`
- `used_token_slots`
- `wasted_token_slots`
- `utilization`
- `allocation_reduction_ratio`

## 面试解释

Paged KV Cache 的重点不是单个 token 算得更快，而是提高 serving 场景下 KV cache 内存利用率。

可以这样说：

> 在 dynamic batching 中，每个请求的上下文长度和完成时间都不同。如果为每个请求预留最大长度的连续 KV cache，短请求会浪费大量显存。Paged KV Cache 把 KV memory 切成 blocks，并用 block table 管理逻辑 token 到物理 block 的映射，请求完成后 blocks 可以被释放和复用。这是 vLLM PagedAttention 能提高 serving capacity 的核心原因之一。

