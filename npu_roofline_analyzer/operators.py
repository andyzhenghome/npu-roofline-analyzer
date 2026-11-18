"""
算子注册表与计算量/内存访问量分析
支持 Transformer 等大模型常用算子
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from enum import Enum
import math


class OperatorCategory(Enum):
    """算子类别"""
    GEMM = "gemm"                    # 矩阵乘法
    ATTENTION = "attention"          # 注意力机制
    ACTIVATION = "activation"        # 激活函数
    NORMALIZATION = "normalization"  # 归一化
    ELEMENTWISE = "elementwise"      # 逐元素操作
    REDUCTION = "reduction"          # 归约操作
    CONVOLUTION = "convolution"      # 卷积
    POOLING = "pooling"              # 池化
    MEMORY = "memory"                # 内存操作 (transpose, reshape, copy)
    COMMUNICATION = "communication"  # 通信算子 (allreduce, allgather)
    CUSTOM = "custom"                # 自定义


@dataclass
class TensorSpec:
    """张量规格"""
    shape: List[int]
    dtype: str = "fp16"
    name: str = ""
    
    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)
    
    @property
    def size_bytes(self) -> int:
        dtype_sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 1, "fp8": 1}
        return self.num_elements * dtype_sizes.get(self.dtype.lower(), 2)


@dataclass
class OperatorInfo:
    """算子分析结果"""
    name: str
    category: OperatorCategory
    flops: int                           # 总浮点操作数
    memory_bytes: int                    # 总内存访问字节数 (读+写)
    arithmetic_intensity: float          # 计算强度 (FLOPS/Byte)
    input_tensors: List[TensorSpec]
    output_tensors: List[TensorSpec]
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def operational_intensity(self) -> float:
        """运算强度 = FLOPs / Bytes"""
        if self.memory_bytes == 0:
            return float('inf')
        return self.flops / self.memory_bytes


class OperatorAnalyzer(ABC):
    """算子分析器基类"""
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> OperatorInfo:
        pass
    
    @abstractmethod
    def get_category(self) -> OperatorCategory:
        pass


class OperatorRegistry:
    """算子注册表 - 单例模式"""
    
    _instance = None
    _analyzers: Dict[str, OperatorAnalyzer] = {}
    _aliases: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_builtin_analyzers()
        return cls._instance
    
    def _init_builtin_analyzers(self):
        """注册内置算子分析器"""
        from .operators_gemm import GemmAnalyzer
        from .operators_attention import AttentionAnalyzer
        from .operators_activation import ActivationAnalyzer
        from .operators_normalization import NormalizationAnalyzer
        from .operators_elementwise import ElementwiseAnalyzer
        from .operators_reduction import ReductionAnalyzer
        from .operators_memory import MemoryOpAnalyzer
        from .operators_communication import CommunicationAnalyzer
        
        self.register("gemm", GemmAnalyzer())
        self.register("matmul", GemmAnalyzer())
        self.register("linear", GemmAnalyzer())
        
        self.register("attention", AttentionAnalyzer())
        self.register("self_attention", AttentionAnalyzer())
        self.register("multihead_attention", AttentionAnalyzer())
        self.register("flash_attention", AttentionAnalyzer())
        
        self.register("gelu", ActivationAnalyzer("gelu"))
        self.register("relu", ActivationAnalyzer("relu"))
        self.register("silu", ActivationAnalyzer("silu"))
        self.register("swish", ActivationAnalyzer("swish"))
        
        self.register("layernorm", NormalizationAnalyzer("layernorm"))
        self.register("rmsnorm", NormalizationAnalyzer("rmsnorm"))
        
        self.register("add", ElementwiseAnalyzer("add"))
        self.register("mul", ElementwiseAnalyzer("mul"))
        self.register("div", ElementwiseAnalyzer("div"))
        self.register("sub", ElementwiseAnalyzer("sub"))
        
        self.register("sum", ReductionAnalyzer("sum"))
        self.register("mean", ReductionAnalyzer("mean"))
        self.register("max", ReductionAnalyzer("max"))
        
        self.register("transpose", MemoryOpAnalyzer("transpose"))
        self.register("reshape", MemoryOpAnalyzer("reshape"))
        self.register("view", MemoryOpAnalyzer("view"))
        self.register("copy", MemoryOpAnalyzer("copy"))
        
        self.register("allreduce", CommunicationAnalyzer("allreduce"))
        self.register("allgather", CommunicationAnalyzer("allgather"))
        self.register("reducescatter", CommunicationAnalyzer("reducescatter"))
    
    def register(self, name: str, analyzer: OperatorAnalyzer):
        """注册算子分析器"""
        self._analyzers[name.lower()] = analyzer
        # 自动注册类别作为别名
        self._aliases[analyzer.get_category().value] = name.lower()
    
    def register_alias(self, alias: str, canonical_name: str):
        """注册别名"""
        self._aliases[alias.lower()] = canonical_name.lower()
    
    def get_analyzer(self, op_name: str) -> Optional[OperatorAnalyzer]:
        """获取算子分析器"""
        canonical = self._aliases.get(op_name.lower(), op_name.lower())
        return self._analyzers.get(canonical)
    
    def analyze(self, op_name: str, *args, **kwargs) -> Optional[OperatorInfo]:
        """分析算子"""
        analyzer = self.get_analyzer(op_name)
        if analyzer is None:
            return None
        return analyzer.analyze(*args, **kwargs)
    
    def list_operators(self) -> List[str]:
        """列出所有已注册算子"""
        return list(set(list(self._analyzers.keys()) + list(self._aliases.keys())))
    
    def list_by_category(self, category: OperatorCategory) -> List[str]:
        """按类别列出算子"""
        return [
            name for name, analyzer in self._analyzers.items()
            if analyzer.get_category() == category
        ]


# 全局注册表实例
registry = OperatorRegistry()


def register_operator(name: str, analyzer: OperatorAnalyzer):
    """注册自定义算子的便捷函数"""
    registry.register(name, analyzer)


def analyze_operator(op_name: str, *args, **kwargs) -> Optional[OperatorInfo]:
    """分析算子的便捷函数"""
    return registry.analyze(op_name, *args, **kwargs)