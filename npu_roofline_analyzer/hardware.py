"""
NPU 硬件规格定义
定义 NPU 的峰值算力、内存带宽、缓存层级等关键参数
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class DataType(Enum):
    """支持的数据类型"""
    FP32 = ("fp32", 4)
    FP16 = ("fp16", 2)
    BF16 = ("bf16", 2)
    INT8 = ("int8", 1)
    INT4 = ("int4", 0.5)

    def __init__(self, name: str, bytes_per_element: float):
        self.type_name = name
        self.bytes = bytes_per_element

    @property
    def size_bytes(self) -> float:
        return self.bytes


class ComputeUnit(Enum):
    """计算单元类型"""
    MAC = "mac"           # Multiply-Accumulate
    TENSOR_CORE = "tensor_core"
    VECTOR_ALU = "vector_alu"
    SCALAR_ALU = "scalar_alu"


@dataclass
class MemoryLevel:
    """存储层级定义"""
    name: str                    # 如: "L1", "L2", "HBM", "DDR"
    size_bytes: int              # 容量
    bandwidth_gb_s: float        # 峰值带宽 (GB/s)
    latency_cycles: int          # 访问延迟 (周期)
    is_on_chip: bool = True      # 是否片上存储


@dataclass
class NPUHardwareSpec:
    """
    NPU 硬件规格完整定义
    
    示例:
        # 类 H100 配置
        spec = NPUHardwareSpec(
            name="NPU-H100-Like",
            peak_tflops={"fp16": 989, "fp32": 67, "bf16": 989, "int8": 1979},
            memory_bandwidth_gb_s=3350,
            compute_units={"tensor_core": 528, "cuda_core": 16896},
            clock_ghz=1.59,
            memory_levels=[
                MemoryLevel("L1", 256*1024, 15000, 4, True),
                MemoryLevel("L2", 50*1024*1024, 5000, 20, True),
                MemoryLevel("HBM", 80*1024*1024*1024, 3350, 100, False),
            ]
        )
    """
    name: str
    peak_tflops: Dict[str, float]                    # 各精度峰值算力 (TFLOPS)
    memory_bandwidth_gb_s: float                     # 显存/主存带宽 (GB/s)
    compute_units: Dict[str, int]                    # 计算单元数量
    clock_ghz: float                                 # 核心频率 (GHz)
    memory_levels: list[MemoryLevel] = field(default_factory=list)
    on_chip_sram_bytes: int = 0                      # 片上 SRAM 总大小
    tensor_core_config: Optional[Dict] = None        # Tensor Core 详细配置
    sparse_support: bool = False                     # 是否支持稀疏计算
    max_cluster_size: int = 1                        # 最大集群规模 (多卡)

    def __post_init__(self):
        # 自动计算片上存储总量
        if self.on_chip_sram_bytes == 0:
            self.on_chip_sram_bytes = sum(
                m.size_bytes for m in self.memory_levels if m.is_on_chip
            )

    def get_peak_tflops(self, dtype: str) -> float:
        """获取指定精度的峰值算力"""
        return self.peak_tflops.get(dtype.lower(), 0.0)

    def get_memory_bandwidth(self, level: str = "dram") -> float:
        """获取指定存储层级的带宽"""
        if level.lower() in ("dram", "hbm", "ddr", "global"):
            return self.memory_bandwidth_gb_s
        for mem in self.memory_levels:
            if mem.name.lower() == level.lower():
                return mem.bandwidth_gb_s
        return self.memory_bandwidth_gb_s

    def get_roofline_ceilings(self, dtype: str = "fp16") -> Dict[str, float]:
        """
        获取 Roofline 模型的各条性能上界 (GFLOPS)
        返回: {层级名: 峰值性能}
        """
        ceilings = {}
        peak = self.get_peak_tflops(dtype) * 1000  # 转为 GFLOPS
        
        # 计算上界
        ceilings["Compute"] = peak
        
        # 内存上界 (按层级)
        for mem in self.memory_levels:
            bw = mem.bandwidth_gb_s  # GB/s
            ceilings[f"Memory_{mem.name}"] = bw * 1000 / self._get_bytes_per_op(dtype)
        
        # 全局内存上界
        ceilings["Memory_DRAM"] = self.memory_bandwidth_gb_s * 1000 / self._get_bytes_per_op(dtype)
        
        return ceilings

    def _get_bytes_per_op(self, dtype: str) -> float:
        """获取每次操作的字节数 (用于内存上界计算)"""
        dt = DataType[dtype.upper()] if dtype.upper() in DataType.__members__ else DataType.FP16
        return dt.size_bytes * 2  # 假设每次操作读写各一次

    @classmethod
    def from_preset(cls, preset: str) -> "NPUHardwareSpec":
        """从预设配置创建"""
        presets = {
            "h100": cls(
                name="NVIDIA H100",
                peak_tflops={"fp16": 989, "fp32": 67, "bf16": 989, "int8": 1979, "fp8": 1979},
                memory_bandwidth_gb_s=3350,
                compute_units={"tensor_core": 528, "cuda_core": 16896},
                clock_ghz=1.59,
                memory_levels=[
                    MemoryLevel("L1", 256*1024, 15000, 4, True),
                    MemoryLevel("L2", 50*1024*1024, 5000, 20, True),
                    MemoryLevel("HBM3", 80*1024*1024*1024, 3350, 100, False),
                ],
                tensor_core_config={"mma_m": 16, "mma_n": 8, "mma_k": 16, "stages": 4},
                sparse_support=True,
            ),
            "a100": cls(
                name="NVIDIA A100",
                peak_tflops={"fp16": 312, "fp32": 19.5, "bf16": 312, "int8": 624},
                memory_bandwidth_gb_s=1555,
                compute_units={"tensor_core": 432, "cuda_core": 6912},
                clock_ghz=1.41,
                memory_levels=[
                    MemoryLevel("L1", 192*1024, 12000, 4, True),
                    MemoryLevel("L2", 40*1024*1024, 4000, 20, True),
                    MemoryLevel("HBM2e", 40*1024*1024*1024, 1555, 100, False),
                ],
                tensor_core_config={"mma_m": 16, "mma_n": 8, "mma_k": 16, "stages": 4},
                sparse_support=True,
            ),
            "ascend910b": cls(
                name="Huawei Ascend 910B",
                peak_tflops={"fp16": 358, "fp32": 44.8, "bf16": 358, "int8": 716},
                memory_bandwidth_gb_s=1200,
                compute_units={"daicore": 256, "vector": 256, "scalar": 256},
                clock_ghz=1.2,
                memory_levels=[
                    MemoryLevel("L1", 512*1024, 8000, 4, True),
                    MemoryLevel("L2", 32*1024*1024, 3000, 20, True),
                    MemoryLevel("HBM2e", 64*1024*1024*1024, 1200, 100, False),
                ],
            ),
            "generic_npu": cls(
                name="Generic NPU",
                peak_tflops={"fp16": 100, "fp32": 25, "bf16": 100, "int8": 400},
                memory_bandwidth_gb_s=512,
                compute_units={"mac": 1024},
                clock_ghz=1.0,
                memory_levels=[
                    MemoryLevel("L1", 128*1024, 5000, 4, True),
                    MemoryLevel("L2", 16*1024*1024, 2000, 20, True),
                    MemoryLevel("DDR", 32*1024*1024*1024, 512, 100, False),
                ],
            ),
        }
        if preset.lower() not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset.lower()]


# 预导出常用预设
H100_SPEC = NPUHardwareSpec.from_preset("h100")
A100_SPEC = NPUHardwareSpec.from_preset("a100")
ASCEND910B_SPEC = NPUHardwareSpec.from_preset("ascend910b")
GENERIC_NPU_SPEC = NPUHardwareSpec.from_preset("generic_npu")