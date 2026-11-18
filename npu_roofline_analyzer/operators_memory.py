"""
内存操作算子分析器 (Transpose, Reshape, View, Copy 等)
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class MemoryOpAnalyzer(OperatorAnalyzer):
    """内存操作分析器 (零 FLOPs，纯内存搬运)"""
    
    def __init__(self, op_type: str):
        self.op_type = op_type.lower()
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.MEMORY
    
    def analyze(
        self,
        input_shape: List[int],
        output_shape: List[int] = None,
        dtype: str = "fp16",
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim
        
        if output_shape is None:
            output_shape = input_shape
        
        output_elements = 1
        for dim in output_shape:
            output_elements *= dim
        
        # 纯内存操作: 0 FLOPs
        total_flops = 0
        
        # 内存: 读输入 + 写输出
        total_memory = (input_elements + output_elements) * bytes_per_elem
        
        input_tensors = [TensorSpec(input_shape, dtype, "input")]
        output_tensors = [TensorSpec(output_shape, dtype, "output")]
        
        return OperatorInfo(
            name=self.op_type,
            category=OperatorCategory.MEMORY,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=0.0,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "input_shape": input_shape,
                "output_shape": output_shape,
                "dtype": dtype,
            }
        )


__all__ = ["MemoryOpAnalyzer"]