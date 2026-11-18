"""
NPU Roofline 静态分析工具
用于大模型算子的计算强度分析与性能上界建模
"""

__version__ = "1.0.0"
__author__ = "NPU Performance Team"

from .analyzer import NPURooflineAnalyzer, RooflineAnalysisResult, RooflinePoint
from .hardware import NPUHardwareSpec, DataType, ComputeUnit, MemoryLevel
from .operators import (
    OperatorRegistry, register_operator, registry,
    OperatorCategory, OperatorInfo, TensorSpec
)
from .models import ModelParser, PyTorchModelParser, ONNXModelParser, HFModelParser, create_parser
from .visualizer import RooflineVisualizer

__all__ = [
    "NPURooflineAnalyzer",
    "RooflineAnalysisResult",
    "RooflinePoint",
    "NPUHardwareSpec",
    "DataType",
    "ComputeUnit",
    "MemoryLevel",
    "OperatorRegistry",
    "register_operator",
    "registry",
    "OperatorCategory",
    "OperatorInfo",
    "TensorSpec",
    "ModelParser",
    "PyTorchModelParser",
    "ONNXModelParser",
    "HFModelParser",
    "create_parser",
    "RooflineVisualizer",
]