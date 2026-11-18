"""
归一化算子分析器 (LayerNorm, RMSNorm)
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class NormalizationAnalyzer(OperatorAnalyzer):
    """归一化算子分析器"""
    
    def __init__(self, norm_type: str):
        self.norm_type = norm_type.lower()
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.NORMALIZATION
    
    def analyze(
        self,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        dtype: str = "fp16",
        eps: float = 1e-5,
        has_weight: bool = True,
        has_bias: bool = True,
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        num_elements = batch * seq_len * hidden_dim
        num_features = hidden_dim
        
        if self.norm_type == "layernorm":
            # LayerNorm: mean, var, normalize, scale, shift
            # FLOPs per feature: sum(x) + sum(x^2) + (x-mean)/sqrt(var+eps) * weight + bias
            # 约: 2*D (mean+var) + 5*D (normalize+scale+shift) = 7*D per token
            flops_per_token = 7 * num_features
        elif self.norm_type == "rmsnorm":
            # RMSNorm: sqrt(mean(x^2)), normalize, scale
            # 约: 2*D (sum sq + mean) + 3*D (normalize+scale) = 5*D per token
            flops_per_token = 5 * num_features
        else:
            flops_per_token = 6 * num_features
        
        total_flops = batch * seq_len * flops_per_token
        
        # 内存访问
        # 读输入 + 读weight/bias + 写输出 + 中间统计量
        size_input = num_elements * bytes_per_elem
        size_weight = num_features * bytes_per_elem if has_weight else 0
        size_bias = num_features * bytes_per_elem if has_bias else 0
        size_output = num_elements * bytes_per_elem
        # 中间统计量 (mean, var/rms): 2 * batch * seq_len * 4bytes (fp32累加)
        size_stats = 2 * batch * seq_len * 4
        
        total_memory = size_input + size_weight + size_bias + size_output + size_stats
        
        input_tensors = [
            TensorSpec([batch, seq_len, hidden_dim], dtype, "input"),
        ]
        if has_weight:
            input_tensors.append(TensorSpec([hidden_dim], dtype, "weight"))
        if has_bias:
            input_tensors.append(TensorSpec([hidden_dim], dtype, "bias"))
        
        output_tensors = [TensorSpec([batch, seq_len, hidden_dim], dtype, "output")]
        
        return OperatorInfo(
            name=self.norm_type,
            category=OperatorCategory.NORMALIZATION,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "batch": batch, "seq_len": seq_len, "hidden_dim": hidden_dim,
                "dtype": dtype, "eps": eps,
                "has_weight": has_weight, "has_bias": has_bias,
            }
        )


__all__ = ["NormalizationAnalyzer"]