"""
Roofline 模型核心分析引擎
计算算子的理论性能上界、实际可达性能、瓶颈分析
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import json

from .hardware import NPUHardwareSpec, DataType
from .operators import OperatorInfo, OperatorCategory, registry


@dataclass
class RooflinePoint:
    """Roofline 模型上的一个点 (算子)"""
    name: str
    category: str
    operational_intensity: float      # FLOPs/Byte
    achieved_gflops: float            # 实际/预测性能 (GFLOPS)
    peak_gflops: float                # 硬件峰值性能 (GFLOPS)
    memory_bandwidth_gb_s: float      # 适用的内存带宽 (GB/s)
    compute_ceiling: float            # 计算上界 (GFLOPS)
    memory_ceiling: float             # 内存上界 (GFLOPS)
    bottleneck: str                   # "compute" | "memory" | "latency" | "bandwidth"
    utilization: float                # 性能利用率 (0-1)
    dtype: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class RooflineAnalysisResult:
    """完整的 Roofline 分析结果"""
    hardware_spec: NPUHardwareSpec
    points: List[RooflinePoint]
    summary: Dict
    warnings: List[str] = field(default_factory=list)


class NPURooflineAnalyzer:
    """NPU Roofline 静态分析器主类"""
    
    def __init__(self, hardware_spec: NPUHardwareSpec, dtype: str = "fp16"):
        """
        初始化分析器
        
        Args:
            hardware_spec: NPU 硬件规格
            dtype: 目标数据类型 (fp16, bf16, fp32, int8, fp8)
        """
        self.hardware = hardware_spec
        self.dtype = dtype.lower()
        self.peak_gflops = hardware_spec.get_peak_tflops(dtype) * 1000
        self.memory_bw_gb_s = hardware_spec.memory_bandwidth_gb_s
        
        # 计算各存储层级的内存上界
        self.memory_ceilings = self._compute_memory_ceilings()
    
    def _compute_memory_ceilings(self) -> Dict[str, float]:
        """计算各存储层级的内存性能上界 (GFLOPS)"""
        ceilings = {}
        bytes_per_op = DataType[self.dtype.upper()].size_bytes * 2 if self.dtype.upper() in DataType.__members__ else 4
        
        # DRAM/HBM 上界
        ceilings["DRAM"] = self.memory_bw_gb_s * 1000 / bytes_per_op
        
        # 片上存储上界
        for mem in self.hardware.memory_levels:
            if mem.is_on_chip:
                ceilings[mem.name] = mem.bandwidth_gb_s * 1000 / bytes_per_op
        
        return ceilings
    
    def analyze_operator(self, op_info: OperatorInfo) -> RooflinePoint:
        """分析单个算子，生成 Roofline 点"""
        
        oi = op_info.operational_intensity
        if oi == float('inf'):
            oi = 1e9  # 纯计算算子
        
        # 确定适用的内存上界 (基于算子特征判断数据驻留层级)
        applicable_bw = self._estimate_applicable_bandwidth(op_info)
        memory_ceiling = applicable_bw * 1000 / (DataType[self.dtype.upper()].size_bytes * 2)
        
        # 计算上界
        compute_ceiling = self.peak_gflops
        
        # 理论性能 = min(计算上界, 内存上界 * OI)
        theoretical_perf = min(compute_ceiling, memory_ceiling * oi)
        
        # 判断瓶颈
        if theoretical_perf >= compute_ceiling * 0.95:
            bottleneck = "compute"
        elif theoretical_perf >= memory_ceiling * oi * 0.95:
            bottleneck = "memory"
        else:
            bottleneck = "latency"  # 其他因素限制 (同步、调度等)
        
        # 估算实际可达性能 (引入效率因子)
        achieved_perf = self._estimate_achieved_performance(
            op_info, theoretical_perf, bottleneck
        )
        
        utilization = achieved_perf / self.peak_gflops if self.peak_gflops > 0 else 0
        
        return RooflinePoint(
            name=op_info.name,
            category=op_info.category.value,
            operational_intensity=oi,
            achieved_gflops=achieved_perf,
            peak_gflops=self.peak_gflops,
            memory_bandwidth_gb_s=applicable_bw,
            compute_ceiling=compute_ceiling,
            memory_ceiling=memory_ceiling,
            bottleneck=bottleneck,
            utilization=utilization,
            dtype=self.dtype,
            metadata={
                "flops": op_info.flops,
                "memory_bytes": op_info.memory_bytes,
                "input_shapes": [t.shape for t in op_info.input_tensors],
                "output_shapes": [t.shape for t in op_info.output_tensors],
                **op_info.metadata,
            }
        )
    
    def _estimate_applicable_bandwidth(self, op_info: OperatorInfo) -> float:
        """估算算子实际使用的内存带宽层级"""
        # 启发式判断数据驻留层级
        total_data_bytes = op_info.memory_bytes
        
        # 按数据量判断可能的缓存层级
        l1_size = 0
        l2_size = 0
        for mem in self.hardware.memory_levels:
            if mem.name.upper() == "L1":
                l1_size = mem.size_bytes
            elif mem.name.upper() == "L2":
                l2_size = mem.size_bytes
        
        # 简化模型：按数据复用程度判断
        # Attention/GEMM 等高复用算子倾向于使用片上存储
        if op_info.category in (OperatorCategory.GEMM, OperatorCategory.ATTENTION):
            # 如果工作集能放入 L2，假设主要走 L2
            working_set = sum(t.size_bytes for t in op_info.input_tensors + op_info.output_tensors)
            if working_set <= l2_size * 0.8:
                for mem in self.hardware.memory_levels:
                    if mem.name.upper() == "L2":
                        return mem.bandwidth_gb_s
            elif working_set <= l1_size * 0.8:
                for mem in self.hardware.memory_levels:
                    if mem.name.upper() == "L1":
                        return mem.bandwidth_gb_s
        
        # 默认走 DRAM
        return self.memory_bw_gb_s
    
    def _estimate_achieved_performance(
        self, 
        op_info: OperatorInfo, 
        theoretical_perf: float,
        bottleneck: str
    ) -> float:
        """估算实际可达性能 (考虑各种效率损失)"""
        
        # 基础效率因子
        base_efficiency = 0.85  # 理论峰值的 85% 为上限
        
        # 根据算子类型调整
        category_efficiency = {
            OperatorCategory.GEMM: 0.95,           # GEMM 高度优化，接近峰值
            OperatorCategory.ATTENTION: 0.75,      # Attention 复杂，有软件开销
            OperatorCategory.ACTIVATION: 0.60,     # 点操作，内存受限严重
            OperatorCategory.NORMALIZATION: 0.55,  # 归约操作多
            OperatorCategory.ELEMENTWISE: 0.50,    # 极度内存受限
            OperatorCategory.REDUCTION: 0.45,      # 同步开销大
            OperatorCategory.MEMORY: 0.90,         # 纯内存搬运，带宽利用率高
            OperatorCategory.COMMUNICATION: 0.70,  # 通信重叠不完全
        }
        
        eff = category_efficiency.get(op_info.category, 0.65)
        
        # 根据瓶颈类型进一步调整
        if bottleneck == "compute":
            # 计算受限：受限于指令吞发、占用率
            eff *= 0.9
        elif bottleneck == "memory":
            # 内存受限：受限于缓存命中率、预取效率
            eff *= 0.8
        elif bottleneck == "latency":
            # 延迟受限：同步、调度、尾部效应
            eff *= 0.7
        
        # 规模效应：小问题规模无法填满硬件
        scale_factor = self._compute_scale_factor(op_info)
        eff *= scale_factor
        
        return theoretical_perf * eff
    
    def _compute_scale_factor(self, op_info: OperatorInfo) -> float:
        """计算规模因子：小问题无法充分利用并行单元"""
        # 简化模型：基于总 FLOPs 估算
        total_flops = op_info.flops
        num_compute_units = sum(self.hardware.compute_units.values())
        
        # 每个计算单元需要一定工作量才能饱和
        flops_per_unit = total_flops / num_compute_units if num_compute_units > 0 else 1e9
        
        # 经验阈值：每单元 1M FLOPs 以上接近饱和
        if flops_per_unit >= 1e6:
            return 1.0
        elif flops_per_unit >= 1e5:
            return 0.8
        elif flops_per_unit >= 1e4:
            return 0.6
        else:
            return 0.4
    
    def analyze_model(self, model_ops: List[OperatorInfo]) -> RooflineAnalysisResult:
        """分析整个模型 (算子序列)"""
        points = [self.analyze_operator(op) for op in model_ops]
        
        # 汇总统计
        total_flops = sum(op.flops for op in model_ops)
        total_memory = sum(op.memory_bytes for op in model_ops)
        
        # 逐算子耗时: 计算类用 FLOPs/性能；纯搬运类(通信)用其自身通信时延
        compute_time_s = 0.0
        comm_time_s = 0.0
        for op, p in zip(model_ops, points):
            if op.flops > 0 and p.achieved_gflops > 0:
                compute_time_s += op.flops / (p.achieved_gflops * 1e9)
            elif op.category == OperatorCategory.COMMUNICATION:
                comm_time_s += op.params.get("comm_time_us", 0.0) * 1e-6
            else:
                # 纯内存搬运算子: 用带宽下界估算
                if op.memory_bytes > 0 and self.memory_bw_gb_s > 0:
                    compute_time_s += op.memory_bytes / (self.memory_bw_gb_s * 1e9)
        total_time_est = compute_time_s + comm_time_s
        
        compute_bound = sum(1 for p in points if p.bottleneck == "compute")
        memory_bound = sum(1 for p in points if p.bottleneck == "memory")
        
        avg_utilization = sum(p.utilization for p in points) / len(points) if points else 0
        
        summary = {
            "total_operators": len(model_ops),
            "total_flops": total_flops,
            "total_memory_bytes": total_memory,
            "overall_arithmetic_intensity": total_flops / total_memory if total_memory > 0 else 0,
            "estimated_total_time_us": total_time_est * 1e6,
            "estimated_compute_time_us": compute_time_s * 1e6,
            "estimated_comm_time_us": comm_time_s * 1e6,
            "compute_bound_ops": compute_bound,
            "memory_bound_ops": memory_bound,
            "latency_bound_ops": len(points) - compute_bound - memory_bound,
            "average_utilization": avg_utilization,
            "peak_gflops": self.peak_gflops,
            "memory_bandwidth_gb_s": self.memory_bw_gb_s,
            "dtype": self.dtype,
        }
        
        # 生成警告
        warnings = []
        if avg_utilization < 0.3:
            warnings.append(f"整体利用率低 ({avg_utilization:.1%})，建议优化算子融合或增加批次大小")
        if memory_bound > compute_bound * 2:
            warnings.append("大部分算子受内存带宽限制，考虑启用 FlashAttention、算子融合或使用更高带宽内存")
        if any(p.operational_intensity < 1 for p in points):
            low_oi_ops = [p.name for p in points if p.operational_intensity < 1]
            warnings.append(f"存在低计算强度算子 (OI<1): {low_oi_ops}，极度受内存带宽限制")
        
        return RooflineAnalysisResult(
            hardware_spec=self.hardware,
            points=points,
            summary=summary,
            warnings=warnings,
        )
    
    def analyze_by_name(self, op_name: str, *args, **kwargs) -> Optional[RooflinePoint]:
        """按算子名称直接分析"""
        op_info = registry.analyze(op_name, *args, **kwargs)
        if op_info is None:
            return None
        return self.analyze_operator(op_info)
    
    def export_json(self, result: RooflineAnalysisResult, filepath: str):
        """导出 JSON 格式结果"""
        data = {
            "hardware": {
                "name": result.hardware_spec.name,
                "peak_tflops": result.hardware_spec.peak_tflops,
                "memory_bandwidth_gb_s": result.hardware_spec.memory_bandwidth_gb_s,
                "dtype": self.dtype,
            },
            "summary": result.summary,
            "warnings": result.warnings,
            "operators": [
                {
                    "name": p.name,
                    "category": p.category,
                    "operational_intensity": p.operational_intensity,
                    "achieved_gflops": p.achieved_gflops,
                    "peak_gflops": p.peak_gflops,
                    "memory_bandwidth_gb_s": p.memory_bandwidth_gb_s,
                    "compute_ceiling_gflops": p.compute_ceiling,
                    "memory_ceiling_gflops": p.memory_ceiling,
                    "bottleneck": p.bottleneck,
                    "utilization": p.utilization,
                    "metadata": p.metadata,
                }
                for p in result.points
            ],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_csv(self, result: RooflineAnalysisResult, filepath: str):
        """导出 CSV 格式结果"""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Operator", "Category", "Operational_Intensity",
                "Achieved_GFLOPS", "Peak_GFLOPS", "Memory_BW_GBps",
                "Compute_Ceiling_GFLOPS", "Memory_Ceiling_GFLOPS",
                "Bottleneck", "Utilization"
            ])
            for p in result.points:
                writer.writerow([
                    p.name, p.category, p.operational_intensity,
                    p.achieved_gflops, p.peak_gflops, p.memory_bandwidth_gb_s,
                    p.compute_ceiling, p.memory_ceiling,
                    p.bottleneck, p.utilization
                ])


__all__ = ["NPURooflineAnalyzer", "RooflinePoint", "RooflineAnalysisResult"]