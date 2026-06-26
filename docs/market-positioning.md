# 市场定位

## 目标职位

TurboInfer 与以下职位最为相关：

- AI Infrastructure Engineer
- LLM Inference Engineer
- ML Systems Engineer
- Model Serving Engineer
- GPU Performance Engineer（Triton/CUDA 密集方向）

## 展示的技能

**系统**：

- 请求调度
- 批处理
- 内存管理
- 延迟分解
- 服务 API 设计
- 基准测试设计

**LLM 推理**：

- Prefill 和 Decode 分离
- KV Cache
- 分页 KV Cache
- 吞吐-延迟权衡
- 量化
- 与 vLLM 风格服务的对比

**GPU**：

- PyTorch 性能分析
- Triton 内核基础
- 内存带宽意识
- 硬件特定基准测试解读

**Infra**：

- 基于 Docker 的环境
- API 服务器
- 指标采集
- 可复现的基准测试配置
- 云 GPU 租赁部署说明

## 简历角度

项目描述示例：

> 构建了 TurboInfer，一个逐步实现 KV Cache、连续批处理、分页 KV Cache、精选 Triton 内核和仅权重量化的迷你 LLM 推理服务引擎。设计了可复现的基准测试，测量 TTFT、TPOT、吞吐、P95 延迟和峰值 GPU 内存，并在匹配的工作负载下与 vLLM 对比结果。

## 面试话题

- 为什么朴素自回归解码浪费算力
- 为什么 KV Cache 将瓶颈转向内存
- 为什么 Prefill 和 Decode 有不同的性能特征
- 连续批处理如何提升 GPU 利用率
- 为什么批处理可能增加尾延迟
- 为什么分页 KV Cache 有助于服务负载
- 为什么 T4 基准测试结果有用但不是最终结论
- 如何设计与 vLLM 的公平对比

## 风险区域

- 过早尝试支持太多模型家族
- 在服务循环可测量之前花费太多时间在内核上
- 在没有足够工作负载细节的情况下报告基准测试数据
- 将 Colab T4 结果视为生产级证据
- 构建面试中无法清晰解释的功能

## 推荐项目边界

第一个完整版本应支持一到两个小型因果语言模型、一个服务 API 和一个清晰的基准测试套件。深度和可测量性比广泛的模型兼容性更重要。
