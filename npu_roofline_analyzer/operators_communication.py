"""
通信算子分析器 (AllReduce, AllGather, ReduceScatter 等)
用于多卡/多核并行场景
"""

from typing import List
from .operators import (
    OperatorAnalyzer, OperatorCategory, OperatorInfo, TensorSpec
)


class CommunicationAnalyzer(OperatorAnalyzer):
    """通信算子分析器"""
    
    def __init__(self, comm_type: str):
        self.comm_type = comm_type.lower()
        # 通信算子的 FLOPs 通常为 0 (纯数据搬运)，但包含归约计算
        self._has_reduction = comm_type.lower() in ("allreduce", "reducescatter")
    
    def get_category(self) -> OperatorCategory:
        return OperatorCategory.COMMUNICATION
    
    def analyze(
        self,
        num_elements: int,
        dtype: str = "fp16",
        world_size: int = 8,          # 参与通信的设备数
        bandwidth_gb_s: float = 200,  # 互联带宽 (GB/s, 如 NVLink, RoCE)
        latency_us: float = 1.0,      # 通信延迟 (us)
        algorithm: str = "ring",      # 通信算法: ring, tree, butterfly
    ) -> OperatorInfo:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
        bytes_per_elem = dtype_sizes.get(dtype.lower(), 2)
        
        data_size_bytes = num_elements * bytes_per_elem
        
        # FLOPs: 只有归约类通信有计算
        if self._has_reduction:
            # AllReduce: 每个元素参与 (world_size-1) 次加法
            # ReduceScatter: 类似
            total_flops = num_elements * (world_size - 1)
        else:
            # AllGather, Broadcast, AllToAll: 纯数据搬运
            total_flops = 0
        
        # 通信量估算 (基于算法)
        if algorithm == "ring":
            # Ring AllReduce: 2 * (world_size - 1) / world_size * data_size
            # Ring AllGather: (world_size - 1) / world_size * data_size
            if self.comm_type == "allreduce":
                comm_volume = 2 * (world_size - 1) / world_size * data_size_bytes
            elif self.comm_type == "allgather":
                comm_volume = (world_size - 1) / world_size * data_size_bytes * world_size
            elif self.comm_type == "reducescatter":
                comm_volume = (world_size - 1) / world_size * data_size_bytes
            else:
                comm_volume = data_size_bytes * (world_size - 1)
        elif algorithm == "tree":
            # Tree 算法: log2(world_size) 步
            steps = math.ceil(math.log2(world_size))
            comm_volume = steps * data_size_bytes
        else:
            comm_volume = data_size_bytes * (world_size - 1)
        
        # 总内存访问 = 本地读写 + 通信量
        # 本地: 读输入缓冲 + 写输出缓冲
        local_memory = 2 * data_size_bytes
        total_memory = local_memory + comm_volume
        
        # 理论通信时间 (us)
        # bandwidth_gb_s 以 GB/s 计, 需先换算为 B/s: *1e9
        comm_time_us = latency_us + (comm_volume / (bandwidth_gb_s * 1e9)) * 1e6
        
        input_tensors = [TensorSpec([num_elements], dtype, "input")]
        output_tensors = [TensorSpec([num_elements], dtype, "output")]
        
        return OperatorInfo(
            name=self.comm_type,
            category=OperatorCategory.COMMUNICATION,
            flops=total_flops,
            memory_bytes=int(total_memory),
            arithmetic_intensity=total_flops / total_memory if total_memory > 0 else float('inf'),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            params={
                "num_elements": num_elements,
                "dtype": dtype,
                "world_size": world_size,
                "bandwidth_gb_s": bandwidth_gb_s,
                "latency_us": latency_us,
                "algorithm": algorithm,
                "comm_volume_bytes": int(comm_volume),
                "comm_time_us": comm_time_us,
            },
            metadata={
                "is_compute_bound": total_flops > 0 and total_flops / total_memory > 100,
                "comm_efficiency": data_size_bytes / comm_volume if comm_volume > 0 else 1.0,
            }
        )


import math
__all__ = ["CommunicationAnalyzer"]