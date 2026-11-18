# NPU Roofline 静态分析工具

专为大模型算子设计的 NPU 性能分析工具，基于 Roofline 模型提供计算强度分析、性能上界建模、瓶颈识别和优化建议。

## ✨ 特性

- **🎯 面向大模型**: 内置 Transformer、Attention、MoE、KV Cache 等大模型核心算子模型
- **🏗️ 硬件建模**: 预设 H100、A100、Ascend 910B 等主流 NPU/GPU 规格，支持自定义硬件
- **📊 多层级 Roofline**: 同时建模 DRAM、L2、L1 等多级存储带宽上界
- **🔍 瓶颈自动识别**: 自动判断算子是计算受限、内存受限还是延迟受限
- **📈 丰富可视化**: ASCII 终端图表、Plotly 交互图、Matplotlib 代码、Markdown 报表
- **🔌 多框架支持**: 支持 PyTorch FX、ONNX、HuggingFace Transformers 模型解析
- **💻 CLI & API**: 命令行工具与 Python API 双模式使用

## 🚀 快速开始

### 安装

```bash
# 基础安装
pip install npu-roofline-analyzer

# 含可选依赖 (PyTorch/ONNX/HF/绘图)
pip install npu-roofline-analyzer[all]
```

### 命令行使用

```bash
# 列出支持的算子
npu-roofline list-ops

# 列出预设硬件
npu-roofline list-hardware

# 分析单个 GEMM 算子
npu-roofline analyze-op gemm --hardware h100 --dtype fp16 \
    --params M=4096 N=4096 K=4096 batch=1

# 分析 FlashAttention
npu-roofline analyze-op attention --hardware h100 \
    --params batch=1 seq_len=2048 num_heads=32 head_dim=128 use_flash=true

# 分析完整 Transformer 模型 (类 LLaMA)
npu-roofline analyze-model transformer --hardware h100 \
    --output llama_result --plot llama_plot

# 分析推理阶段 (KV Cache)
npu-roofline analyze-model inference --hardware h100

# 分析 MoE 模型
npu-roofline analyze-model moe --hardware h100
```

### Python API 使用

```python
from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
)

# 1. 选择硬件
hardware = NPUHardwareSpec.from_preset("h100")

# 2. 创建分析器
analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
visualizer = RooflineVisualizer(hardware, dtype="fp16")

# 3. 分析单个算子
gemm_op = registry.analyze("gemm", M=4096, N=4096, K=4096, batch=1, dtype="fp16")
point = analyzer.analyze_operator(gemm_op)
print(f"GEMM 性能: {point.achieved_gflops:.1f} GFLOPS, 瓶颈: {point.bottleneck}")

# 4. 分析 Attention
attn_op = registry.analyze("attention", 
    batch=1, seq_len=2048, num_heads=32, head_dim=128,
    dtype="fp16", use_flash=True
)
point = analyzer.analyze_operator(attn_op)
print(f"Attention 性能: {point.achieved_gflops:.1f} GFLOPS")

# 5. 批量分析模型并可视化
# ... 构建算子列表 ...
result = analyzer.analyze_model(ops)
print(visualizer.generate_ascii_plot(result))
print(visualizer.export_summary_table(result))

# 6. 导出绘图数据
plotly_data = visualizer.generate_plotly_data(result)
# 可直接用 plotly 绘制交互图表
```

## 📖 核心概念

### Roofline 模型

Roofline 模型将算子性能上界建模为两条线的包络：
- **计算上界**: 硬件峰值算力 (TFLOPS)
- **内存上界**: 内存带宽 × 计算强度 (FLOPs/Byte)

实际性能 = min(计算上界, 内存上界)

### 计算强度 (Operational Intensity)

```
OI = 总 FLOPs / 总内存访问字节数
```

- OI 高 → 计算受限 → 优化方向: 指令吞发、Tensor Core 利用率
- OI 低 → 内存受限 → 优化方向: 算子融合、缓存复用、数据布局

### 瓶颈类型

| 类型 | 判断条件 | 优化建议 |
|------|----------|----------|
| `compute` | 接近计算上界 | 提升占用率、使用 Tensor Core、混合精度 |
| `memory` | 接近内存上界 | 算子融合、分块、FlashAttention、压缩 |
| `latency` | 远低于两条上界 | 减少同步、流水线并行、异步执行 |

## 🏗️ 支持的算子

| 类别 | 算子 | 关键参数 |
|------|------|----------|
| **GEMM** | `gemm`, `matmul`, `linear` | M, N, K, batch, trans_a/b, has_bias |
| **Attention** | `attention`, `flash_attention`, `multihead_attention` | batch, seq_len, num_heads, head_dim, causal, flash, kv_cache |
| **Activation** | `gelu`, `relu`, `silu`, `swish`, `sigmoid`, `tanh` | num_elements, inplace |
| **Normalization** | `layernorm`, `rmsnorm` | batch, seq_len, hidden_dim, eps |
| **Elementwise** | `add`, `mul`, `div`, `sub`, `fma` | num_elements, num_inputs, broadcast |
| **Reduction** | `sum`, `mean`, `max`, `min` | input_shape, reduce_dims, keepdim |
| **Memory** | `transpose`, `reshape`, `view`, `copy` | input_shape, output_shape |
| **Communication** | `allreduce`, `allgather`, `reducescatter` | num_elements, world_size, bandwidth, algorithm |

## 🎮 预设硬件配置

| 预设 | 说明 | FP16 TFLOPS | 内存带宽 |
|------|------|-------------|----------|
| `h100` | NVIDIA H100 (HBM3) | 989 | 3350 GB/s |
| `a100` | NVIDIA A100 (HBM2e) | 312 | 1555 GB/s |
| `ascend910b` | 华为昇腾 910B | 358 | 1200 GB/s |
| `generic_npu` | 通用 NPU 参考 | 100 | 512 GB/s |

### 自定义硬件

```bash
npu-roofline custom-hardware --name "MyNPU" \
    --peak-tflops '{"fp16": 200, "fp32": 50, "int8": 800}' \
    --bandwidth 800 \
    --compute-units '{"tensor_core": 256, "cuda_core": 8192}' \
    --clock 1.5 \
    --output my_npu.json
```

## 📊 输出示例

### PNG Roofline 图（无需 matplotlib）

仓库自带 `render_roofline_png.py`，仅依赖 Pillow，直接渲染 log-log Roofline 图：

```bash
python render_roofline_png.py      # 生成 roofline_mixed.png + demo_roofline_mixed.json
```

图中同时画了两组算子，便于对照两个极端：

| 场景 | 计算强度 | 性能 | 结论 |
|------|----------|------|------|
| Prefill (B=1, L=2048) | OI ≈ 1024–1642 | 668–846 TFLOPS | 计算受限，已接近峰值 |
| Decode (B=1, KV=2048) | OI ≈ 0.77–1.0 | 129–254 TFLOPS | 内存受限，落在 DRAM 斜线上 |

![Roofline 示例](roofline_mixed.png)

要点：**Prefill 是算力问题，Decode 是带宽问题** —— 同一颗芯片，两个阶段的瓶颈完全不同，
优化手段也应不同（Prefill 用更大 tile / 更高精度 Tensor Core；Decode 用 batching、权重压缩、KV Cache 量化）。

### ASCII Roofline 图表

```
                                                                         
  10000 ┤                                                                 
       │                                                                 
  1000 ┤                    ═══════════════════════════════════════════
       │                    Compute Peak: 989000 GFLOPS                  
   100 ┤              ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
       │            ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
    10 ┤        ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
       │      ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
     1 ┤  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
       │┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
       └────────────────────────────────────────────────────────────
        0.01    0.1      1       10      100     1000    10000
                    Operational Intensity (FLOPs/Byte)

图例: ●=Compute-bound  ■=Memory-bound  ▲=Latency-bound
Compute Peak: 989000 GFLOPS  |  DRAM BW: 3350 GB/s  |  Dtype: fp16
```

### Markdown 汇总报表

```markdown
# Roofline 分析报告: NVIDIA H100 (FP16)

## 硬件规格
- **峰值算力**: 989000 GFLOPS (989.0 TFLOPS)
- **内存带宽**: 3350 GB/s
- **数据类型**: FP16

## 算子分析汇总

| 算子 | 类别 | 计算强度 | 实际性能 | 峰值利用率 | 瓶颈 |
|------|------|----------|----------|------------|------|
| layer_0_qkv_proj | gemm | 128.5 | 890123.4 | 90.0% | compute |
| layer_0_attention | attention | 45.2 | 654321.7 | 66.2% | memory |
| layer_0_attn_out | gemm | 128.5 | 890123.4 | 90.0% | compute |
| ... | ... | ... | ... | ... | ... |

## 整体统计
- **总算子数**: 289
- **总 FLOPs**: 1,234,567,890,123
- **平均利用率**: 72.3%
- **计算受限**: 156, **内存受限**: 112, **延迟受限**: 21

## ⚠️ 优化建议
- 存在低计算强度算子 (OI<1): layernorm, add, mul，极度受内存带宽限制
- 建议启用算子融合 (LayerNorm+Residual, SiLU+Mul)
```

## ⚠️ 建模假设与精度说明

本工具是**静态上界估算器**，不对应某次实测，请按如下假设解读结果：

1. **FLOPs 计数规则**：GEMM 为 `2*M*N*K*batch`（乘加各记 1 次）；grouped/batched GEMM 的 `batch`
   即分组数，`group_size` 只影响调度与指针开销，**不重复计入计算量**。
2. **内存访问**：按"每个张量读写各一次"计数，未建模 cache 命中。`use_flash=True` 时才把
   Attention 中间矩阵留在片上，因此 FlashAttention 的 OI 会显著高于标准实现。
3. **内存上界**：`perf = bandwidth × OI`，`bandwidth` 由算子工作集大小启发式选择
   （能装进 L2 用 L2 带宽，否则用 DRAM）。这正是 `D0_AttnOut` 在 OI=1.0 被判为 compute-bound
   的原因——它被假设为 L2 驻留。
4. **效率因子**：按算子类别 × 瓶颈类型 × 问题规模打折（GEMM 0.85–0.95，逐元素 0.4–0.6 等），
   用于从"理论上界"逼近"可达性能"。这些系数是经验值，**换硬件/换编译器需重新标定**。
5. **通信时延**：`latency_us + comm_volume / (bandwidth_GBps × 1e9) × 1e6`，Ring/Tree 两种算法。
   不含拥塞、重传与计算通信重叠。

如果要做真实对标，建议：用 nsys/profiler 采集单算子实测 TFLOPS 与 DRAM 吞吐，替换
`analyzer.py` 中的 `_estimate_achieved_performance` 与 `_estimate_applicable_bandwidth` 经验式。

## 🧪 测试

```bash
python -m pytest tests/ -v      # 11 个用例
python verify_install.py        # 端到端冒烟验证
python demo.py                  # 4 组场景演示并导出 JSON/CSV/绘图数据
```

测试覆盖：硬件预设、GEMM/Attention 计算量、模型级汇总、可视化输出、CLI 导入，以及
四个回归点（grouped GEMM 重复计数、通信时延单位、通信算子耗时统计、OI=0 时绘图不崩溃）。

## 🔧 扩展开发

### 添加自定义算子

```python
from npu_roofline_analyzer import OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec, register_operator

class MyCustomOpAnalyzer(OperatorAnalyzer):
    def get_category(self):
        return OperatorCategory.CUSTOM
    
    def analyze(self, param1: int, param2: float, dtype: str = "fp16"):
        # 计算 FLOPs 和内存访问
        flops = param1 * param2 * 2
        memory = param1 * 4 + param2 * 4  # 简化
        
        return OperatorInfo(
            name="my_custom_op",
            category=OperatorCategory.CUSTOM,
            flops=flops,
            memory_bytes=memory,
            arithmetic_intensity=flops/memory,
            input_tensors=[TensorSpec([param1], dtype)],
            output_tensors=[TensorSpec([param2], dtype)],
            params={"param1": param1, "param2": param2}
        )

# 注册
register_operator("my_custom_op", MyCustomOpAnalyzer())

# 使用
op_info = registry.analyze("my_custom_op", param1=1024, param2=2048)
```

## 📁 项目结构

```
npu_roofline_analyzer/
├── __init__.py           # 主包导出
├── hardware.py           # NPU 硬件规格定义
├── operators.py          # 算子注册表与基类
├── operators_gemm.py     # GEMM 族算子分析
├── operators_attention.py # Attention 族算子分析
├── operators_activation.py # 激活函数分析
├── operators_normalization.py # 归一化分析
├── operators_elementwise.py # 逐元素操作分析
├── operators_reduction.py   # 归约操作分析
├── operators_memory.py      # 内存操作分析
├── operators_communication.py # 通信算子分析
├── analyzer.py           # Roofline 核心分析引擎
├── models.py             # 模型解析器
├── visualizer.py         # 可视化输出
├── cli.py                # 命令行入口
└── examples/
    └── transformer_analysis.py  # Transformer 完整示例

根目录:
├── demo.py               # 4 组场景演示 (LLaMA / 推理 / MoE / 多硬件对比)
├── verify_install.py     # 冒烟验证
├── render_roofline_png.py # Pillow 渲染 Roofline PNG
└── tests/test_basic.py   # 11 个单元 + 回归测试
```

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发环境

```bash
git clone https://github.com/your-org/npu-roofline-analyzer
cd npu-roofline-analyzer
pip install -e .[dev]
pytest tests/
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- Roofline 模型原论文: Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- NVIDIA/AMD/华为等厂商公开的硬件规格文档
- 开源社区的各类性能分析工具