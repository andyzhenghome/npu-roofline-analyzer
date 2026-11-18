"""
Attention 算子分析器
支持标准 Attention、FlashAttention、Multi-Head Attention 等变体
"""

from typing import List, Optional
import math
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class AttentionAnalyzer(OperatorAnalyzer):
    """Attention 算子分析器"""
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.ATTENTION
    
    def analyze(
        self,
        batch: int,           # batch size
        seq_len: int,         # 序列长度 (Q)
        seq_len_kv: int = None,  # KV 序列长度 (默认等于 seq_len)
        num_heads: int = 32,  # 注意力头数
        head_dim: int = 128,  # 每个头的维度
        dtype: str = "fp16",
        causal: bool = False,     # 是否因果掩码
        use_flash: bool = False,  # 是否使用 FlashAttention
        kv_cache: bool = False,   # 是否使用 KV Cache (推理场景)
        dropout: float = 0.0,     # Dropout 概率
        window_size: int = -1,    # 滑动窗口注意力 (-1 表示全局)
        alibi: bool = False,      # ALiBi 位置编码
    ) -> OperatorInfo:
        """
        分析 Attention 算子
        
        标准 Attention 流程:
        1. QKV 投影: 3 * B * L * (H*D) * (H*D)  (Linear/GEMM)
        2. Q @ K^T: B * H * L * L * D
        3. Softmax: B * H * L * L
        4. Attn @ V: B * H * L * L * D
        5. Output 投影: B * L * (H*D) * (H*D)
        
        FlashAttention 通过分块减少 HBM 读写，但 FLOPs 相同
        """
        if seq_len_kv is None:
            seq_len_kv = seq_len
        
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        hidden_dim = num_heads * head_dim
        Lq, Lkv = seq_len, seq_len_kv
        B, H, D = batch, num_heads, head_dim
        
        # === FLOPs 计算 ===
        
        # 1. QKV 投影 (3 个 Linear: input[B,L,H*D] @ W[H*D,H*D])
        flops_qkv = 3 * 2 * B * Lq * hidden_dim * hidden_dim
        
        # 2. Q @ K^T: [B, H, Lq, D] @ [B, H, D, Lkv] -> [B, H, Lq, Lkv]
        flops_qkt = 2 * B * H * Lq * Lkv * D
        
        # 3. Softmax (近似: 5 FLOPs/element: max, sub, exp, sum, div)
        flops_softmax = 5 * B * H * Lq * Lkv
        
        # 4. Attn @ V: [B, H, Lq, Lkv] @ [B, H, Lkv, D] -> [B, H, Lq, D]
        flops_attn_v = 2 * B * H * Lq * Lkv * D
        
        # 5. Output 投影: [B, Lq, H*D] @ [H*D, H*D]
        flops_out = 2 * B * Lq * hidden_dim * hidden_dim
        
        # Dropout (训练时)
        flops_dropout = B * H * Lq * Lkv if dropout > 0 else 0
        
        # KV Cache 场景: 只计算新 token 的 Q @ K^T
        if kv_cache:
            # 假设增量推理: Lq=1 (单步), Lkv=当前序列长度
            flops_qkt = 2 * B * H * 1 * Lkv * D
            flops_softmax = 5 * B * H * 1 * Lkv
            flops_attn_v = 2 * B * H * 1 * Lkv * D
            flops_qkv = 3 * 2 * B * 1 * hidden_dim * hidden_dim  # 只计算新 token 的 QKV
            flops_out = 2 * B * 1 * hidden_dim * hidden_dim
            flops_dropout = 0  # 推理无 dropout
        
        total_flops = (flops_qkv + flops_qkt + flops_softmax + 
                       flops_attn_v + flops_out + flops_dropout)
        
        # === 内存访问计算 ===
        
        # 输入: [B, Lq, H*D]
        size_input = B * Lq * hidden_dim * bytes_per_elem
        
        # QKV 权重: 3 * [H*D, H*D]
        size_qkv_weight = 3 * hidden_dim * hidden_dim * bytes_per_elem
        
        # QKV 输出 (中间结果，FlashAttention 可避免写回 HBM)
        size_qkv_output = 3 * B * Lq * hidden_dim * bytes_per_elem
        
        # Attention 矩阵: [B, H, Lq, Lkv] (标准 Attention 需要 materialize)
        if use_flash:
            # FlashAttention: 分块计算，不需要存完整 Attention 矩阵
            # 只需存储当前块的 S, P, O
            block_size = min(128, Lq)  # 典型分块大小
            size_attn_matrix = B * H * block_size * Lkv * bytes_per_elem * 2  # S + P
        else:
            size_attn_matrix = B * H * Lq * Lkv * bytes_per_elem
        
        # V: [B, Lkv, H*D] (来自 QKV 输出)
        size_v = B * Lkv * hidden_dim * bytes_per_elem
        
        # Output 投影权重: [H*D, H*D]
        size_out_weight = hidden_dim * hidden_dim * bytes_per_elem
        
        # Output: [B, Lq, H*D]
        size_output = B * Lq * hidden_dim * bytes_per_elem
        
        # KV Cache 读写 (推理场景)
        size_kv_cache = 0
        if kv_cache:
            # 读取历史 K/V: 2 * B * (Lkv-1) * H * D
            # 写入新 K/V: 2 * B * 1 * H * D
            size_kv_cache = 2 * B * Lkv * hidden_dim * bytes_per_elem * 2  # 读+写
        
        # 总内存访问
        if use_flash:
            # FlashAttention: Q, K, V 各读一次，O 写一次，中间结果留在 SRAM
            total_memory = (size_input + size_qkv_weight + 
                           size_qkv_output * 0.5 +  # 部分融合
                           size_v + size_out_weight + size_output + size_kv_cache)
        else:
            # 标准 Attention: 所有中间结果都要读写 HBM
            total_memory = (size_input + size_qkv_weight + size_qkv_output +
                           size_attn_matrix * 2 +  # Attention 矩阵读写
                           size_v + size_out_weight + size_output + size_kv_cache)
        
        # === 构建张量规格 ===
        input_tensors = [
            TensorSpec([B, Lq, hidden_dim], dtype, "input"),
            TensorSpec([3, hidden_dim, hidden_dim], dtype, "qkv_weight"),
        ]
        if kv_cache:
            input_tensors.append(TensorSpec([2, B, Lkv, H, D], dtype, "kv_cache"))
        
        output_tensors = [TensorSpec([B, Lq, hidden_dim], dtype, "output")]
        
        return OperatorInfo(
            name="attention" if not use_flash else "flash_attention",
            category=OperatorCategory.ATTENTION,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "batch": B, "seq_len": Lq, "seq_len_kv": Lkv,
                "num_heads": H, "head_dim": D, "hidden_dim": hidden_dim,
                "causal": causal, "flash": use_flash, "kv_cache": kv_cache,
                "dropout": dropout, "window_size": window_size, "alibi": alibi,
                "dtype": dtype,
            },
            metadata={
                "flops_breakdown": {
                    "qkv_proj": flops_qkv,
                    "qkt": flops_qkt,
                    "softmax": flops_softmax,
                    "attn_v": flops_attn_v,
                    "out_proj": flops_out,
                    "dropout": flops_dropout,
                },
                "memory_breakdown": {
                    "input": size_input,
                    "qkv_weight": size_qkv_weight,
                    "qkv_output": size_qkv_output,
                    "attn_matrix": size_attn_matrix,
                    "v": size_v,
                    "out_weight": size_out_weight,
                    "output": size_output,
                    "kv_cache": size_kv_cache,
                },
                "is_memory_bound": total_flops / total_memory < 100 if total_memory > 0 else False,
            }
        )


# 导出
__all__ = ["AttentionAnalyzer"]