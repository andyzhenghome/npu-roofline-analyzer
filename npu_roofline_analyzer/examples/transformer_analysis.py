"""
Transformer 模型完整 Roofline 分析示例
演示如何使用工具分析大模型各层性能
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
    OperatorCategory,
)


def analyze_llama_like_model():
    """分析类 LLaMA 模型 (Decoder-only Transformer)"""
    
    # ========== 1. 选择硬件平台 ==========
    # 可选: "h100", "a100", "ascend910b", "generic_npu"
    hardware = NPUHardwareSpec.from_preset("h100")
    print(f"硬件平台: {hardware.name}")
    print(f"峰值 FP16 算力: {hardware.peak_tflops['fp16']} TFLOPS")
    print(f"内存带宽: {hardware.memory_bandwidth_gb_s} GB/s")
    print()
    
    # ========== 2. 创建分析器 ==========
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    # ========== 3. 定义模型配置 (类 LLaMA-7B) ==========
    config = {
        "batch_size": 1,
        "seq_len": 2048,          # 预填充阶段序列长度
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "num_layers": 32,
        "intermediate_dim": 11008,  # FFN 中间层 (通常 2.7x hidden)
        "vocab_size": 32000,
        "dtype": "fp16",
    }
    
    B, L, H, D = config["batch_size"], config["seq_len"], config["hidden_dim"], config["head_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    inter_dim = config["intermediate_dim"]
    
    print(f"模型配置: LLaMA-like ({num_layers} layers, {H}d, {num_heads}h)")
    print(f"Batch={B}, SeqLen={L}")
    print()
    
    # ========== 4. 逐层分析 ==========
    all_ops = []
    
    for layer_idx in range(num_layers):
        prefix = f"layer_{layer_idx}"
        
        # --- Attention 模块 ---
        # QKV 投影: 3 * [B, L, H] @ [H, H] -> 3 * [B, L, H]
        qkv_op = registry.analyze("gemm", 
            M=B*L, N=3*H, K=H, batch=1, dtype="fp16"
        )
        qkv_op.name = f"{prefix}_qkv_proj"
        all_ops.append(qkv_op)
        
        # Attention 核心 (FlashAttention)
        attn_op = registry.analyze("attention",
            batch=B, seq_len=L, num_heads=num_heads, head_dim=D,
            dtype="fp16", use_flash=True
        )
        attn_op.name = f"{prefix}_attention"
        all_ops.append(attn_op)
        
        # Attention Output 投影: [B, L, H] @ [H, H]
        attn_out = registry.analyze("gemm",
            M=B*L, N=H, K=H, batch=1, dtype="fp16"
        )
        attn_out.name = f"{prefix}_attn_out"
        all_ops.append(attn_out)
        
        # LayerNorm 1 (Pre-LN)
        ln1 = registry.analyze("layernorm",
            batch=B, seq_len=L, hidden_dim=H, dtype="fp16"
        )
        ln1.name = f"{prefix}_ln1"
        all_ops.append(ln1)
        
        # --- FFN 模块 (SwiGLU: Gate * Up -> Down) ---
        # Gate Proj: [B, L, H] @ [H, inter_dim]
        gate_proj = registry.analyze("gemm",
            M=B*L, N=inter_dim, K=H, batch=1, dtype="fp16"
        )
        gate_proj.name = f"{prefix}_gate_proj"
        all_ops.append(gate_proj)
        
        # Up Proj: [B, L, H] @ [H, inter_dim]
        up_proj = registry.analyze("gemm",
            M=B*L, N=inter_dim, K=H, batch=1, dtype="fp16"
        )
        up_proj.name = f"{prefix}_up_proj"
        all_ops.append(up_proj)
        
        # SiLU 激活
        silu = registry.analyze("silu",
            num_elements=B*L*inter_dim, dtype="fp16"
        )
        silu.name = f"{prefix}_silu"
        all_ops.append(silu)
        
        # Element-wise Mul (Gate * Up)
        mul = registry.analyze("mul",
            num_elements=B*L*inter_dim, dtype="fp16"
        )
        mul.name = f"{prefix}_mul"
        all_ops.append(mul)
        
        # Down Proj: [B, L, inter_dim] @ [inter_dim, H]
        down_proj = registry.analyze("gemm",
            M=B*L, N=H, K=inter_dim, batch=1, dtype="fp16"
        )
        down_proj.name = f"{prefix}_down_proj"
        all_ops.append(down_proj)
        
        # LayerNorm 2
        ln2 = registry.analyze("layernorm",
            batch=B, seq_len=L, hidden_dim=H, dtype="fp16"
        )
        ln2.name = f"{prefix}_ln2"
        all_ops.append(ln2)
        
        # Residual Add (2次: attn残差 + ffn残差)
        for i, name in enumerate(["attn_residual", "ffn_residual"]):
            add_op = registry.analyze("add",
                num_elements=B*L*H, dtype="fp16", inplace=True
            )
            add_op.name = f"{prefix}_{name}"
            all_ops.append(add_op)
    
    # --- Embedding & Output Layer ---
    # Input Embedding (Lookup, 视为 Memory Op)
    embed = registry.analyze("gemm",  # 近似为 GEMM
        M=B*L, N=H, K=config["vocab_size"], batch=1, dtype="fp16", trans_b=True
    )
    embed.name = "input_embedding"
    all_ops.append(embed)
    
    # Output Projection (LM Head): [B, L, H] @ [H, Vocab]
    lm_head = registry.analyze("gemm",
        M=B*L, N=config["vocab_size"], K=H, batch=1, dtype="fp16"
    )
    lm_head.name = "lm_head"
    all_ops.append(lm_head)
    
    print(f"总共分析算子数: {len(all_ops)}")
    print()
    
    # ========== 5. 执行 Roofline 分析 ==========
    result = analyzer.analyze_model(all_ops)
    
    # ========== 6. 输出结果 ==========
    
    # 6.1 ASCII 图表
    print("=" * 60)
    print("ROOFLINE ASCII 图表")
    print("=" * 60)
    ascii_plot = visualizer.generate_ascii_plot(result, width=100, height=30)
    print(ascii_plot)
    print()
    
    # 6.2 汇总表格
    print("=" * 60)
    print("分析汇总表格")
    print("=" * 60)
    summary_table = visualizer.export_summary_table(result)
    # 替换无法在 GBK 终端显示的字符
    safe_table = summary_table.replace('⚠️', '[WARN]').replace('●', '[*]').replace('■', '[#]').replace('▲', '[^]')
    print(safe_table)
    print()
    
    # 6.3 导出文件
    analyzer.export_json(result, "llama_roofline.json")
    analyzer.export_csv(result, "llama_roofline.csv")
    print("已导出: llama_roofline.json, llama_roofline.csv")
    
    # 6.4 生成绘图代码
    plotly_data = visualizer.generate_plotly_data(result)
    with open("llama_roofline_plotly.json", "w") as f:
        json.dump(plotly_data, f, indent=2)
    print("已导出 Plotly 数据: llama_roofline_plotly.json")
    
    mpl_code = visualizer.generate_matplotlib_code(result)
    with open("llama_roofline_plot.py", "w") as f:
        f.write(mpl_code)
    print("已生成 Matplotlib 绘图脚本: llama_roofline_plot.py")
    
    return result


def analyze_inference_kv_cache():
    """分析推理阶段 (KV Cache) 的 Roofline"""
    from npu_roofline_analyzer import NPUHardwareSpec, NPURooflineAnalyzer, registry
    
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    # 推理配置: 单步解码, 带 KV Cache
    config = {
        "batch_size": 1,
        "seq_len": 1,           # 单步生成
        "kv_seq_len": 2048,     # 当前 KV Cache 长度
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "dtype": "fp16",
    }
    
    B, L, Lkv = config["batch_size"], config["seq_len"], config["kv_seq_len"]
    H, D = config["hidden_dim"], config["head_dim"]
    num_heads = config["num_heads"]
    
    print("=" * 60)
    print("推理阶段 KV Cache Roofline 分析")
    print("=" * 60)
    
    ops = []
    
    # QKV 投影 (只计算新 token)
    qkv = registry.analyze("gemm", M=B*L, N=3*H, K=H, batch=1, dtype="fp16")
    qkv.name = "decode_qkv_proj"
    ops.append(qkv)
    
    # Attention with KV Cache
    attn = registry.analyze("attention",
        batch=B, seq_len=L, seq_len_kv=Lkv, num_heads=num_heads, head_dim=D,
        dtype="fp16", use_flash=True, kv_cache=True
    )
    attn.name = "decode_attention_kvcache"
    ops.append(attn)
    
    # Output Proj
    out = registry.analyze("gemm", M=B*L, N=H, K=H, batch=1, dtype="fp16")
    out.name = "decode_attn_out"
    ops.append(out)
    
    result = analyzer.analyze_model(ops)
    
    print(visualizer.generate_ascii_plot(result, width=100, height=25))
    print()
    safe_table = visualizer.export_summary_table(result).replace('⚠️', '[WARN]').replace('●', '[*]').replace('■', '[#]').replace('▲', '[^]')
    print(safe_table)
    
    return result


def analyze_moe_model():
    """分析 MoE (Mixture of Experts) 模型"""
    from npu_roofline_analyzer import NPUHardwareSpec, NPURooflineAnalyzer, registry
    
    hardware = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hardware, dtype="fp16")
    visualizer = RooflineVisualizer(hardware, dtype="fp16")
    
    # MoE 配置 (类 Mixtral)
    config = {
        "batch_size": 4,
        "seq_len": 2048,
        "hidden_dim": 4096,
        "num_experts": 8,
        "top_k": 2,
        "expert_dim": 14336,  # 每个专家的 FFN 维度
        "dtype": "fp16",
    }
    
    B, L, H = config["batch_size"], config["seq_len"], config["hidden_dim"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    expert_dim = config["expert_dim"]
    
    print("=" * 60)
    print("MoE 模型 Roofline 分析")
    print("=" * 60)
    
    ops = []
    
    # Router (Gate): [B*L, H] @ [H, num_experts] -> [B*L, num_experts]
    router = registry.analyze("gemm", M=B*L, N=num_experts, K=H, batch=1, dtype="fp16")
    router.name = "moe_router"
    ops.append(router)
    
    # Expert 并行计算 (Grouped GEMM)
    # 实际上每个 token 只送到 top_k 个专家
    tokens_per_expert = (B * L * top_k) // num_experts
    
    # Gate Proj (Grouped): 所有专家并行
    gate = registry.analyze("gemm", 
        M=tokens_per_expert, N=expert_dim, K=H, 
        batch=num_experts, group_size=num_experts, dtype="fp16"
    )
    gate.name = "moe_gate_proj_grouped"
    ops.append(gate)
    
    # Up Proj (Grouped)
    up = registry.analyze("gemm",
        M=tokens_per_expert, N=expert_dim, K=H,
        batch=num_experts, group_size=num_experts, dtype="fp16"
    )
    up.name = "moe_up_proj_grouped"
    ops.append(up)
    
    # SiLU + Mul
    act = registry.analyze("silu", num_elements=num_experts * tokens_per_expert * expert_dim, dtype="fp16")
    act.name = "moe_silu"
    ops.append(act)
    
    mul = registry.analyze("mul", num_elements=num_experts * tokens_per_expert * expert_dim, dtype="fp16")
    mul.name = "moe_mul"
    ops.append(mul)
    
    # Down Proj (Grouped)
    down = registry.analyze("gemm",
        M=tokens_per_expert, N=H, K=expert_dim,
        batch=num_experts, group_size=num_experts, dtype="fp16"
    )
    down.name = "moe_down_proj_grouped"
    ops.append(down)
    
    # All-to-all 通信 (专家并行间)
    alltoall = registry.analyze("allgather",
        num_elements=B*L*H, dtype="fp16", world_size=num_experts, 
        bandwidth_gb_s=600, algorithm="ring"  # NVLink 带宽
    )
    alltoall.name = "moe_alltoall"
    ops.append(alltoall)
    
    result = analyzer.analyze_model(ops)
    
    print(visualizer.generate_ascii_plot(result, width=100, height=25))
    print()
    safe_table = visualizer.export_summary_table(result).replace('⚠️', '[WARN]').replace('●', '[*]').replace('■', '[#]').replace('▲', '[^]')
    print(safe_table)
    
    return result


if __name__ == "__main__":
    import json
    
    print("NPU Roofline 静态分析工具 - Transformer 示例")
    print("=" * 60)
    
    # 1. 训练/预填充阶段完整模型分析
    result1 = analyze_llama_like_model()
    
    print("\n" + "="*60 + "\n")
    
    # 2. 推理阶段 KV Cache 分析
    result2 = analyze_inference_kv_cache()
    
    print("\n" + "="*60 + "\n")
    
    # 3. MoE 模型分析
    result3 = analyze_moe_model()