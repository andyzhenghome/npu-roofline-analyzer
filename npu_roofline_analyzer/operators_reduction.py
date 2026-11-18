"""
归约操作算子分析器 (Sum, Mean, Max 等)
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class ReductionAnalyzer(OperatorAnalyzer):
    """归约操作分析器"""
    
    def __init__(self, reduce_type: str):
        self.reduce_type = reduce_type.lower()
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.REDUCTION
    
    def analyze(
        self,
        input_shape: List[int],
        reduce_dims: List[int],
        dtype: str = "fp16",
        keepdim: bool = False,
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        
        # 归约后的元素数
        output_shape = [input_shape[i] for i in range(len(input_shape)) if i not in reduce_dims]
        if keepdim:
            output_shape = [1 if i in reduce_dims else input_shape[i] for i in range(len(input_shape))]
        
        output_elements = 1
        for dim in output_shape:
            output_elements *= dim
        
        reduced_elements = total_elements // output_elements
        
        # FLOPs: 每个归约元素参与一次操作
        if self.reduce_type in ("sum", "mean"):
            flops_per_reduction = reduced_elements  # 加法
        elif self.reduce_type in ("max", "min"):
            flops_per_reduction = reduced_elements  # 比较
        else:
            flops_per_reduction = reduced_elements
        
        total_flops = output_elements * flops_per_reduction
        
        # 内存: 读输入 + 写输出
        total_memory = total_elements * bytes_per_elem + output_elements * bytes_per_elem
        
        input_tensors = [TensorSpec(input_shape, dtype, "input")]
        output_tensors = [TensorSpec(output_shape, dtype, "output")]
        
        return OperatorInfo(
            name=self.reduce_type,
            category=OperatorCategory.REDUCTION,
            flops=total_flops,
            memory_bytes=total_memory,
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "input_shape": input_shape,
                "reduce_dims": reduce_dims,
                "keepdim": keepdim,
                "dtype": dtype,
                "reduced_elements": reduced_elements,
            }
        )


__all__ = ["ReductionAnalyzer"]