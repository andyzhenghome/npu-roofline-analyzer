#!/usr/bin/env python3
"""
快速验证脚本：运行基础功能测试
"""

import sys
sys.path.insert(0, '.')

from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
)


def main():
    print("=" * 60)
    print("NPU Roofline Analyzer - 快速验证")
    print("=" * 60)
    
    # 1. 测试硬件预设
    print("\n1. 硬件预设测试")
    for preset in ["h100", "a100", "ascend910b", "generic_npu"]:
        hw = NPUHardwareSpec.from_preset(preset)
        print(f"  {preset:15s}: {hw.name} | FP16: {hw.peak_tflops['fp16']} TFLOPS | BW: {hw.memory_bandwidth_gb_s} GB/s")
    
    # 2. 测试算子注册表
    print("\n2. 算子注册表测试")
    ops = registry.list_operators()
    print(f"  已注册算子数: {len(ops)}")
    for op in sorted(set(ops))[:10]:
        print(f"    - {op}")
    print(f"    ... 共 {len(ops)} 个")
    
    # 3. 测试单算子分析
    print("\n3. 单算子分析测试 (GEMM 4096x4096x4096)")
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    
    gemm = registry.analyze("gemm", M=4096, N=4096, K=4096, batch=1, dtype="fp16")
    point = analyzer.analyze_operator(gemm)
    print(f"  FLOPs: {gemm.flops:,}")
    print(f"  Memory: {gemm.memory_bytes:,} bytes")
    print(f"  OI: {gemm.operational_intensity:.2f} FLOPs/Byte")
    print(f"  Predicted Perf: {point.achieved_gflops:.0f} GFLOPS")
    print(f"  Bottleneck: {point.bottleneck}")
    print(f"  Utilization: {point.utilization:.1%}")
    
    # 4. 测试 FlashAttention
    print("\n4. FlashAttention 分析测试")
    attn = registry.analyze("attention", 
        batch=1, seq_len=2048, num_heads=32, head_dim=128,
        dtype="fp16", use_flash=True)
    point = analyzer.analyze_operator(attn)
    print(f"  FLOPs: {attn.flops:,}")
    print(f"  Memory: {attn.memory_bytes:,} bytes")
    print(f"  OI: {attn.operational_intensity:.2f} FLOPs/Byte")
    print(f"  Predicted Perf: {point.achieved_gflops:.0f} GFLOPS")
    print(f"  Bottleneck: {point.bottleneck}")
    
    # 5. 测试模型级分析
    print("\n5. 迷你模型分析测试 (2层 Transformer)")
    ops = []
    for i in range(2):
        ops.append(registry.analyze("gemm", M=1*1024, N=4096, K=4096, batch=1, dtype="fp16"))  # QKV
        ops.append(registry.analyze("attention", batch=1, seq_len=1024, num_heads=32, head_dim=128, dtype="fp16", use_flash=True))
        ops.append(registry.analyze("gemm", M=1*1024, N=4096, K=4096, batch=1, dtype="fp16"))  # Out
        ops.append(registry.analyze("gemm", M=1*1024, N=11008, K=4096, batch=1, dtype="fp16"))  # FFN Up
        ops.append(registry.analyze("silu", num_elements=1*1024*11008, dtype="fp16"))
        ops.append(registry.analyze("gemm", M=1*1024, N=4096, K=11008, batch=1, dtype="fp16"))  # FFN Down
    
    result = analyzer.analyze_model(ops)
    print(f"  总算子数: {result.summary['total_operators']}")
    print(f"  总 FLOPs: {result.summary['total_flops']:,}")
    print(f"  平均利用率: {result.summary['average_utilization']:.1%}")
    print(f"  计算受限: {result.summary['compute_bound_ops']}")
    print(f"  内存受限: {result.summary['memory_bound_ops']}")
    print(f"  建议: {len(result.warnings)} 条")
    for w in result.warnings:
        print(f"    - {w}")
    
    # 6. 测试可视化
    print("\n6. 可视化输出测试")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    # ASCII 图表 (简化版)
    ascii_plot = visualizer.generate_ascii_plot(result, width=80, height=20)
    print("  ASCII 图表生成成功 (前20行):")
    for line in ascii_plot.split('\n')[:20]:
        print(f"    {line}")
    
    # Markdown 表格
    table = visualizer.export_summary_table(result)
    print("\n  Markdown 表格生成成功 (前15行):")
    for line in table.split('\n')[:15]:
        print(f"    {line}")
    
    print("\n" + "=" * 60)
    print("[OK] 所有验证通过！工具可正常使用。")
    print("=" * 60)
    print("\n下一步:")
    print("  python -m npu_roofline_analyzer.examples.transformer_analysis  # 完整示例")
    print("  npu-roofline --help  # CLI 使用帮助")
    print("  pip install -e .[all]  # 安装完整依赖")


if __name__ == "__main__":
    main()