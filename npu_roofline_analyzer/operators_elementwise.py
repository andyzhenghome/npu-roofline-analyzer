"""
逐元素操作算子分析器 (Add, Mul, Div, Sub 等)
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class ElementwiseAnalyzer(OperatorAnalyzer):
    """逐元素操作分析器"""
    
    def __init__(self, op_type: str):
        self.op_type = op_type.lower()
        self._flops_per_element = {
            "add": 1, "sub": 1, "mul": 1, "div": 1,
            "fma": 2,   # fused multiply-add
            "max": 1, "min": 1,
            "pow": 8, "sqrt": 4, "rsqrt": 4,
        }
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.ELEMENTWISE
    
    def analyze(
        self,
        num_elements: int,
        dtype: str = "fp16",
        num_inputs: int = 2,        # 输入张量数量
        broadcast: bool = False,    # 是否有广播
        inplace: bool = False,
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        flops_per_elem = self._flops_per_element.get(self.op_type, 1)
        total_flops = num_elements * flops_per_elem
        
        # 内存访问
        if inplace:
            # inplace: 读取一个输入修改后写回
            total_memory = 2 * num_elements * bytes_per_elem
        else:
            # 读取所有输入 + 写输出
            total_memory = (num_inputs + 1) * num_elements * bytes_per_elem
        
        # 广播场景: 广播张量元素数较少
        if broadcast:
            total_memory = num_elements * bytes_per_elem * (1 + 1) + num_elements * bytes_per_elem
        
        input_tensors = [TensorSpec([num_elements], dtype, f"input_{i}") for i in range(num_inputs)]
        output_tensors = [TensorSpec([num_elements], dtype, "output")]
        
        return OperatorInfo(
            name=self.op_type,
            category=OperatorCategory.ELEMENTWISE,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "num_elements": num_elements,
                "dtype": dtype,
                "num_inputs": num_inputs,
                "broadcast": broadcast,
                "inplace": inplace,
            }
        )


__all__ = ["ElementwiseAnalyzer"]