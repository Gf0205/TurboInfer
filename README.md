# TurboInfer

TurboInfer 是一个面向学习的 LLM 推理基础设施项目。目标是构建一个小型但可量化的推理引擎，复现现代推理服务系统（如 vLLM、SGLang、TensorRT-LLM）的核心思想。

本项目并非要替代这些生产级系统。它的价值在于让每个优化都变得可见、可测量、可解释：

- 模型加载与朴素自回归解码
- KV Cache 与 Prefill/Decode 分离
- 连续批处理与请求调度
- 分页 KV Cache / 基于块的内存管理
- 精选 Triton 内核
- 仅权重量化
- OpenAI 兼容的服务 API
- 与 vLLM 的基准对比测试

## 项目定位

本项目面向 AI Infrastructure、LLM Inference、ML Systems 等职位。聚焦于这些职位通常关注的工程问题：

- 如何测量和分解延迟？
- 为什么 KV Cache 能提升解码效率，又引入了多少内存开销？
- 为什么 Prefill 和 Decode 有所不同？
- 批处理如何提升吞吐，同时影响尾延迟？
- 服务系统除了模型执行还需要什么？
- 在受控工作负载下，自定义引擎能多接近 vLLM？

## 硬件计划

第一阶段仅使用 Google Colab T4 进行功能验证和小模型基准测试。T4 的结果不应被视为最终性能证据，因为它是较老的 GPU，不能代表更新的推理硬件。

推荐阶段：

1. 本地 CPU：仓库结构、API 设计、文档、试运行测试
2. Google Colab T4：使用小模型进行功能验证
3. AutoDL 3090 / 4090 / A10：迭代性能优化工作
4. AutoDL A100 / H20：最终基准测试和报告

推荐模型：

- Qwen2.5-0.5B / 1.5B
- TinyLlama-1.1B
- Llama-3.2-1B（如有条件访问）

当前主要 GPU 验证环境为 AutoDL RTX 3090。推荐工作流是在本地整理代码与文档，通过 GitHub 同步到 AutoDL，再在 AutoDL 上 `git pull` 后运行 CUDA/Triton 基准测试。

## 成功标准

最终项目应包含：

- 一个可运行的推理服务器
- 基准测试脚本和可复现的配置
- 每个主要优化的前后指标对比
- 与 vLLM 的对比结果
- 清晰说明权衡和瓶颈的文档
- 一份适合简历或面试讨论的项目总结

## 仓库结构

实现路线图见 [docs/project-plan.md](docs/project-plan.md)。

## 当前进展

- `naive_no_kv_cache` baseline
- `hf_kv_cache` single-request KV Cache
- `static_batch_kv_cache` fixed-arrival batched decode
- Minimal FastAPI serving API at `/v1/completions`
- Real HTTP continuous batching engine with queued request admission
- Real paged KV metadata allocator with block tables and utilization stats
- vLLM comparison workflow and RTX 3090 report template
- Triton RMSNorm kernel benchmark with PyTorch reference comparison
- Triton SiLU-Mul kernel benchmark with PyTorch reference comparison
- Triton RoPE kernel benchmark with PyTorch reference comparison
- Triton Paged Decode Attention kernel benchmark with paged KV metadata
- PagedKVBuffer integration benchmark covering allocator + real K/V tensor storage + paged attention
- Controlled single-layer paged attention path from Q/K/V projections to paged decode attention
- Qwen2.5/Qwen3 model shape profiles and GQA-aware paged decode attention path
- Performance metrics guide: [docs/performance-metrics.md](docs/performance-metrics.md)
