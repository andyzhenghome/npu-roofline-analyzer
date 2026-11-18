"""
激活函数算子分析器
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class ActivationAnalyzer(OperatorAnalyzer):
    """激活函数分析器"""
    
    def __init__(self, act_type: str):
        self.act_type = act_type.lower()
        self._flops_per_element = {
            "relu": 1,      # max(0, x)
            "gelu": 8,      # 近似: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            "silu": 4,      # x * sigmoid(x)
            "swish": 4,     # 同 silu
            "sigmoid": 4,   # 1 / (1 + exp(-x))
            "tanh": 4,      # tanh(x)
            "elu": 3,       # x if x>0 else exp(x)-1
            "leaky_relu": 1, # max(0.01*x, x)
        }
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.ACTIVATION
    
    def analyze(
        self,
        num_elements: int,
        dtype: str = "fp16",
        inplace: bool = False,
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        flops_per_elem = self._flops_per_element.get(self.act_type, 4)
        total_flops = num_elements * flops_per_elem
        
        # 内存: 读输入 + 写输出 (inplace 只读写一次)
        if inplace:
            total_memory = num_elements * bytes_per_elem  # 读修改写算一次
        else:
            total_memory = 2 * num_elements * bytes_per_elem
        
        input_tensors = [TensorSpec([num_elements], dtype, "input")]
        output_tensors = [TensorSpec([num_elements], dtype, "output")]
        
        return OperatorInfo(
            name=self.act_type,
            category=OperatorCategory.ACTIVATION,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "num_elements": num_elements,
                "dtype": dtype,
                "inplace": inplace,
                "flops_per_element": flops_per_elem,
            }
        )


__all__ = ["ActivationAnalyzer"]