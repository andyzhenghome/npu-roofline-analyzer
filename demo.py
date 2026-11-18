#!/usr/bin/env python3
"""
快速演示脚本：生成 Roofline 图表并保存为文件
"""

import sys
sys.path.insert(0, '.')

# Windows GBK 控制台下的稳健输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
)


def demo_h100_llama():
    """H100 上 LLaMA-7B 预填充阶段分析"""
    print("🔍 分析: H100 + LLaMA-7B 预填充 (seq=2048)")
    
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    # 模型配置
    B, L, H = 1, 2048, 4096
    num_heads, head_dim = 32, 128
    inter_dim = 11008
    num_layers = 32
    
    ops = []
    for layer_idx in range(num_layers):
        prefix = f"L{layer_idx}"
        
        # QKV Proj
        ops.append(registry.analyze("gemm", M=B*L, N=3*H, K=H, batch=1, dtype="fp16"))
        ops[-1].name = f"{prefix}_QKV"
        
        # FlashAttention
        ops.append(registry.analyze("attention", batch=B, seq_len=L, num_heads=num_heads, head_dim=head_dim, dtype="fp16", use_flash=True))
        ops[-1].name = f"{prefix}_Attn"
        
        # Attn Out
        ops.append(registry.analyze("gemm", M=B*L, N=H, K=H, batch=1, dtype="fp16"))
        ops[-1].name = f"{prefix}_AttnOut"
        
        # FFN Gate + Up (SwiGLU)
        ops.append(registry.analyze("gemm", M=B*L, N=inter_dim, K=H, batch=1, dtype="fp16"))
        ops[-1].name = f"{prefix}_Gate"
        ops.append(registry.analyze("gemm", M=B*L, N=inter_dim, K=H, batch=1, dtype="fp16"))
        ops[-1].name = f"{prefix}_Up"
        
        # SiLU + Mul
        ops.append(registry.analyze("silu", num_elements=B*L*inter_dim, dtype="fp16"))
        ops[-1].name = f"{prefix}_SiLU"
        ops.append(registry.analyze("mul", num_elements=B*L*inter_dim, dtype="fp16"))
        ops[-1].name = f"{prefix}_Mul"
        
        # FFN Down
        ops.append(registry.analyze("gemm", M=B*L, N=H, K=inter_dim, batch=1, dtype="fp16"))
        ops[-1].name = f"{prefix}_Down"
    
    result = analyzer.analyze_model(ops)
    
    # 保存结果
    analyzer.export_json(result, "demo_h100_llama.json")
    analyzer.export_csv(result, "demo_h100_llama.csv")
    
    # 生成绘图数据
    plotly = visualizer.generate_plotly_data(result)
    with open("demo_h100_llama_plotly.json", "w") as f:
        json.dump(plotly, f, indent=2)
    
    mpl = visualizer.generate_matplotlib_code(result)
    with open("demo_h100_llama_plot.py", "w") as f:
        f.write(mpl)
    
    # 打印摘要
    print(f"  总算子数: {result.summary['total_operators']}")
    print(f"  总 FLOPs: {result.summary['total_flops']:,}")
    print(f"  平均利用率: {result.summary['average_utilization']:.1%}")
    print(f"  计算受限: {result.summary['compute_bound_ops']}")
    print(f"  内存受限: {result.summary['memory_bound_ops']}")
    print(f"  预估耗时: {result.summary['estimated_total_time_us']:.1f} μs")
    print(f"  ⚠️ 警告: {len(result.warnings)} 条")
    for w in result.warnings:
        print(f"    - {w}")
    
    print("  📁 输出文件:")
    print("    - demo_h100_llama.json (完整数据)")
    print("    - demo_h100_llama.csv (表格数据)")
    print("    - demo_h100_llama_plotly.json (Plotly 交互图)")
    print("    - demo_h100_llama_plot.py (Matplotlib 脚本)")
    print()


def demo_inference():
    """推理阶段 KV Cache 分析"""
    print("🔍 分析: H100 推理阶段 (KV Cache, seq=2048->1)")
    
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    B, L, Lkv = 1, 1, 2048
    H, num_heads, head_dim = 4096, 32, 128
    
    ops = [
        registry.analyze("gemm", M=B*L, N=3*H, K=H, batch=1, dtype="fp16"),
        registry.analyze("attention", batch=B, seq_len=L, seq_len_kv=Lkv, num_heads=num_heads, head_dim=head_dim, dtype="fp16", use_flash=True, kv_cache=True),
        registry.analyze("gemm", M=B*L, N=H, K=H, batch=1, dtype="fp16"),
    ]
    ops[0].name = "Decode_QKV"
    ops[1].name = "Decode_Attn_KVCache"
    ops[2].name = "Decode_Out"
    
    result = analyzer.analyze_model(ops)
    
    analyzer.export_json(result, "demo_inference.json")
    
    print(f"  平均利用率: {result.summary['average_utilization']:.1%}")
    print(f"  预估单步耗时: {result.summary['estimated_total_time_us']:.1f} μs")
    print("  📁 输出: demo_inference.json")
    print()


def demo_moe():
    """MoE 模型分析"""
    print("🔍 分析: H100 + MoE (8 Experts, Top-2)")
    
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    B, L, H = 4, 2048, 4096
    num_experts, top_k, expert_dim = 8, 2, 14336
    tokens_per_expert = (B * L * top_k) // num_experts
    
    ops = [
        registry.analyze("gemm", M=B*L, N=num_experts, K=H, batch=1, dtype="fp16"),  # Router
        registry.analyze("gemm", M=tokens_per_expert, N=expert_dim, K=H, batch=num_experts, group_size=num_experts, dtype="fp16"),  # Gate
        registry.analyze("gemm", M=tokens_per_expert, N=expert_dim, K=H, batch=num_experts, group_size=num_experts, dtype="fp16"),  # Up
        registry.analyze("silu", num_elements=num_experts * tokens_per_expert * expert_dim, dtype="fp16"),
        registry.analyze("mul", num_elements=num_experts * tokens_per_expert * expert_dim, dtype="fp16"),
        registry.analyze("gemm", M=tokens_per_expert, N=H, K=expert_dim, batch=num_experts, group_size=num_experts, dtype="fp16"),  # Down
        registry.analyze("allgather", num_elements=B*L*H, dtype="fp16", world_size=num_experts, bandwidth_gb_s=600),  # All-to-all
    ]
    ops[0].name = "MoE_Router"
    ops[1].name = "MoE_Gate_Grouped"
    ops[2].name = "MoE_Up_Grouped"
    ops[3].name = "MoE_SiLU"
    ops[4].name = "MoE_Mul"
    ops[5].name = "MoE_Down_Grouped"
    ops[6].name = "MoE_AllToAll"
    
    result = analyzer.analyze_model(ops)
    
    analyzer.export_json(result, "demo_moe.json")
    
    print(f"  平均利用率: {result.summary['average_utilization']:.1%}")
    print(f"  通信算子瓶颈: {ops[6].category}")
    print("  📁 输出: demo_moe.json")
    print()


def demo_hardware_comparison():
    """多硬件对比"""
    print("🔍 多硬件对比: 同一 GEMM 在不同 NPU 上的表现")
    
    gemm = registry.analyze("gemm", M=4096, N=4096, K=4096, batch=1, dtype="fp16")
    
    for preset in ["h100", "a100", "ascend910b", "generic_npu"]:
        hw = NPUHardwareSpec.from_preset(preset)
        analyzer = NPURooflineAnalyzer(hw, dtype="fp16")
        point = analyzer.analyze_operator(gemm)
        
        print(f"  {preset:15s}: {point.achieved_gflops:>10.0f} GFLOPS | {point.utilization:>5.1%} | {point.bottleneck}")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("NPU Roofline Analyzer - 快速演示")
    print("=" * 60)
    print()
    
    demo_hardware_comparison()
    demo_h100_llama()
    demo_inference()
    demo_moe()
    
    print("=" * 60)
    print("✅ 演示完成！生成的文件可用于进一步分析。")
    print("=" * 60)