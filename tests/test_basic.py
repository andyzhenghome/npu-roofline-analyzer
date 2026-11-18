# 基础测试

import pytest
from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
    OperatorCategory,
)


def test_hardware_presets():
    """测试预设硬件配置"""
    h100 = NPUHardwareSpec.from_preset("h100")
    assert h100.name == "NVIDIA H100"
    assert h100.peak_tflops["fp16"] == 989
    assert h100.memory_bandwidth_gb_s == 3350
    
    a100 = NPUHardwareSpec.from_preset("a100")
    assert a100.peak_tflops["fp16"] == 312
    
    ascend = NPUHardwareSpec.from_preset("ascend910b")
    assert ascend.name == "Huawei Ascend 910B"


def test_gemm_analyzer():
    """测试 GEMM 分析器"""
    op = registry.analyze("gemm", M=1024, N=1024, K=1024, batch=1, dtype="fp16")
    
    assert op is not None
    assert op.category == OperatorCategory.GEMM
    assert op.flops == 2 * 1024 * 1024 * 1024  # 2*M*N*K
    assert op.memory_bytes > 0
    assert op.operational_intensity > 0


def test_attention_analyzer():
    """测试 Attention 分析器"""
    # 标准 Attention
    op = registry.analyze("attention", 
        batch=1, seq_len=2048, num_heads=32, head_dim=128, dtype="fp16")
    
    assert op is not None
    assert op.category == OperatorCategory.ATTENTION
    assert op.flops > 0
    
    # FlashAttention
    op_flash = registry.analyze("attention",
        batch=1, seq_len=2048, num_heads=32, head_dim=128, 
        dtype="fp16", use_flash=True)
    
    # FlashAttention 内存访问更少
    assert op_flash.memory_bytes < op.memory_bytes


def test_analyzer_basic():
    """测试核心分析器"""
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    
    # 分析一个 GEMM
    gemm_op = registry.analyze("gemm", M=4096, N=4096, K=4096, batch=1, dtype="fp16")
    point = analyzer.analyze_operator(gemm_op)
    
    assert point is not None
    assert point.operational_intensity > 0
    assert point.achieved_gflops > 0
    assert point.peak_gflops == hardware.get_peak_tflops("fp16") * 1000
    assert point.bottleneck in ("compute", "memory", "latency")
    assert 0 <= point.utilization <= 1


def test_model_analysis():
    """测试模型级分析"""
    hardware = NPUHardwareSpec.from_preset("generic_npu")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    
    # 简单模型: Embedding + 2层 Transformer
    ops = []
    
    # Embedding
    ops.append(registry.analyze("gemm", M=1*2048, N=4096, K=32000, batch=1, dtype="fp16", trans_b=True))
    
    for i in range(2):
        # Attention
        ops.append(registry.analyze("attention", batch=1, seq_len=2048, num_heads=32, head_dim=128, dtype="fp16", use_flash=True))
        # FFN
        ops.append(registry.analyze("gemm", M=1*2048, N=11008, K=4096, batch=1, dtype="fp16"))
        ops.append(registry.analyze("silu", num_elements=1*2048*11008, dtype="fp16"))
        ops.append(registry.analyze("gemm", M=1*2048, N=4096, K=11008, batch=1, dtype="fp16"))
    
    result = analyzer.analyze_model(ops)
    
    assert result is not None
    assert len(result.points) == len(ops)
    assert "total_operators" in result.summary
    assert result.summary["total_operators"] == len(ops)
    assert result.summary["average_utilization"] >= 0


def test_visualizer():
    """测试可视化器"""
    hardware = NPUHardwareSpec.from_preset("h100")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    ops = [
        registry.analyze("gemm", M=4096, N=4096, K=4096, batch=1, dtype="fp16"),
        registry.analyze("attention", batch=1, seq_len=2048, num_heads=32, head_dim=128, dtype="fp16", use_flash=True),
    ]
    result = analyzer.analyze_model(ops)
    
    # ASCII 图表
    ascii_plot = visualizer.generate_ascii_plot(result)
    assert "Compute Peak" in ascii_plot
    assert "DRAM BW" in ascii_plot
    
    # Markdown 表格
    table = visualizer.export_summary_table(result)
    assert "Roofline 分析报告" in table
    assert "| 算子 | 类别 |" in table
    
    # Plotly 数据
    plotly = visualizer.generate_plotly_data(result)
    assert "data" in plotly
    assert "layout" in plotly
    assert len(plotly["data"]) > 2  # 上界线 + 数据点


def test_cli_import():
    """测试 CLI 模块可导入"""
    from npu_roofline_analyzer.cli import main
    assert main is not None


def test_grouped_gemm_no_double_count():
    """回归: grouped GEMM 的 FLOPs 不应再乘 group_size (batch 已表示分组数)"""
    base = registry.analyze("gemm", M=1024, N=1024, K=1024,
                            batch=8, group_size=1, dtype="fp16")
    grouped = registry.analyze("gemm", M=1024, N=1024, K=1024,
                               batch=8, group_size=8, dtype="fp16")

    assert base.flops == 2 * 1024 ** 3 * 8
    assert grouped.flops == base.flops, "group_size 不应重复计入计算量"


def test_comm_time_unit():
    """回归: 通信时延按 GB/s 换算, 不能出现 s/ms 量级错误"""
    op = registry.analyze("allgather", num_elements=50_000_000, dtype="fp16",
                          world_size=8, bandwidth_gb_s=600, algorithm="ring", latency_us=1.0)
    # 150MB 有效通信量 / 600GB/s ≈ 0.25ms 量级, 绝不应达到秒级
    assert 1.0 <= op.params["comm_time_us"] < 10_000, op.params["comm_time_us"]


def test_comm_op_in_model_time():
    """回归: 纯通信算子参与模型耗时统计 (不应为 0 或被忽略)"""
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")

    ops = [
        registry.analyze("gemm", M=1024, N=1024, K=1024, batch=1, dtype="fp16"),
        registry.analyze("allgather", num_elements=50_000_000, dtype="fp16",
                         world_size=8, bandwidth_gb_s=600, latency_us=1.0),
    ]
    result = analyzer.analyze_model(ops)
    assert result.summary["estimated_total_time_us"] > 0
    assert result.summary["estimated_comm_time_us"] > 0


def test_zero_oi_points_render():
    """回归: OI=0 的通信算子不应让 ASCII 绘图抛 math domain error"""
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")

    ops = [registry.analyze("allgather", num_elements=1024, dtype="fp16", world_size=8)]
    result = analyzer.analyze_model(ops)
    text = visualizer.generate_ascii_plot(result)
    assert "Compute Peak" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])