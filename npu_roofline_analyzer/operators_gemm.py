"""
GEMM / MatMul / Linear 算子分析器
支持各种矩阵乘法变体及其融合形式
"""

from dataclasses import dataclass
from typing import List, Optional
import math
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class GemmAnalyzer(OperatorAnalyzer):
    """通用 GEMM 分析器: C = A @ B^T + bias (可选)"""
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.GEMM
    
    def analyze(
        self,
        M: int,           # A: [M, K], 输出: [M, N]
        N: int,
        K: int,
        dtype: str = "fp16",
        trans_a: bool = False,
        trans_b: bool = False,
        has_bias: bool = False,
        batch: int = 1,               # Batch GEMM
        group_size: int = 1,          # Grouped GEMM (MoE)
        alpha: float = 1.0,           # 缩放因子
        beta: float = 0.0,            # C 缩放因子 (β=0 表示 C 不参与累加)
    ) -> OperatorInfo:
        """
        分析 GEMM 算子
        
        标准 GEMM: C[M,N] = α * A[M,K] @ B[K,N] + β * C[M,N]
        
        FLOPs = 2 * M * N * K * batch  (乘加各算1次; batch 已表示 GEMM 个数/分组数,
               group_size 仅用于描述分组调度开销, 不重复乘入计算量)
        Memory = read(A) + read(B) + read(C if β≠0) + write(C) + read(bias if has_bias)
        """
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        # 计算 FLOPs
        # 矩阵乘法: M*N*K 次乘加 = 2*M*N*K FLOPs
        # grouped GEMM 中 batch 即分组个数，故不再乘 group_size
        flops_per_gemm = 2 * M * N * K
        total_flops = flops_per_gemm * batch
        
        # 内存访问
        # A: [batch, M, K] 或 [batch, K, M] (trans_a)
        size_a = batch * M * K * bytes_per_elem
        # B: [batch, K, N] 或 [batch, N, K] (trans_b)
        size_b = batch * K * N * bytes_per_elem
        # C: [batch, M, N] 读取 (如果 β != 0) + 写入
        size_c_read = batch * M * N * bytes_per_elem if beta != 0 else 0
        size_c_write = batch * M * N * bytes_per_elem
        # Bias: [N] 或 [M, N] (broadcast)
        size_bias = N * bytes_per_elem if has_bias else 0
        
        # Grouped GEMM 额外开销 (指针数组、偏移量等)
        group_overhead = group_size * 3 * 8 if group_size > 1 else 0  # 简化估算
        
        total_memory = size_a + size_b + size_c_read + size_c_write + size_bias + group_overhead
        
        # 构建张量规格
        input_tensors = [
            TensorSpec([batch, M, K] if not trans_a else [batch, K, M], dtype, "A"),
            TensorSpec([batch, K, N] if not trans_b else [batch, N, K], dtype, "B"),
        ]
        if beta != 0:
            input_tensors.append(TensorSpec([batch, M, N], dtype, "C"))
        if has_bias:
            input_tensors.append(TensorSpec([N], dtype, "bias"))
        
        output_tensors = [TensorSpec([batch, M, N], dtype, "output")]
        
        return OperatorInfo(
            name="gemm",
            category=OperatorCategory.GEMM,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "M": M, "N": N, "K": K,
                "batch": batch, "group_size": group_size,
                "trans_a": trans_a, "trans_b": trans_b,
                "has_bias": has_bias, "alpha": alpha, "beta": beta,
                "dtype": dtype,
            },
            metadata={
                "flops_per_gemm": flops_per_gemm,
                "bytes_per_elem": bytes_per_elem,
            }
        )


class BatchedGemmAnalyzer(GemmAnalyzer):
    """Batched GEMM 分析器"""
    pass  # 继承即可，通过 batch 参数控制


class GroupedGemmAnalyzer(GemmAnalyzer):
    """Grouped GEMM 分析器 (MoE 场景)"""
    pass  # 通过 group_size 参数控制


# 导出
__all__ = ["GemmAnalyzer", "BatchedGemmAnalyzer", "GroupedGemmAnalyzer"]