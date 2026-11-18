"""
Roofline 模型可视化工具
支持 ASCII 图表、JSON 导出、绘图数据生成 (可用于 matplotlib/plotly)
"""

from typing import List, Dict, Optional
import json
import math

from .analyzer import RooflineAnalysisResult, RooflinePoint
from .hardware import NPUHardwareSpec


class RooflineVisualizer:
    """Roofline 可视化器"""
    
    def __init__(self, hardware_spec: NPUHardwareSpec, dtype: str = "fp16"):
        self.hardware = hardware_spec
        self.dtype = dtype
        self.peak_gflops = hardware_spec.get_peak_tflops(dtype) * 1000
        self.memory_bw_gb_s = hardware_spec.memory_bandwidth_gb_s
        
        # 计算内存上界线
        from .hardware import DataType
        bytes_per_op = DataType[dtype.upper()].size_bytes * 2 if dtype.upper() in DataType.__members__ else 4
        self.memory_ceiling_gflops = self.memory_bw_gb_s * 1000 / bytes_per_op
        
        # 片上存储上界
        self.cache_ceilings = {}
        for mem in hardware_spec.memory_levels:
            if mem.is_on_chip:
                self.cache_ceilings[mem.name] = mem.bandwidth_gb_s * 1000 / bytes_per_op
    
    def generate_ascii_plot(
        self, 
        result: RooflineAnalysisResult,
        width: int = 80,
        height: int = 25,
        log_scale: bool = True
    ) -> str:
        """生成 ASCII Roofline 图表"""
        
        points = result.points
        if not points:
            return "无数据点"
        
        # 确定坐标范围
        oi_values = [p.operational_intensity for p in points]
        perf_values = [p.achieved_gflops for p in points]
        
        min_oi = min(oi_values) * 0.5
        max_oi = max(oi_values) * 2
        min_perf = min(perf_values) * 0.5
        max_perf = max(max(perf_values), self.peak_gflops) * 1.2
        
        # 防御: 全零 OI/perf (如纯通信算子集合) 会导致 log 域无定义
        min_oi = max(min_oi, 1e-2)
        max_oi = max(max_oi, min_oi * 10)
        min_perf = max(min_perf, 1e-2)
        max_perf = max(max_perf, min_perf * 10)
        
        if log_scale:
            min_oi = max(min_oi, 0.01)
            min_perf = max(min_perf, 0.01)
            x_scale = lambda x: math.log10(x)
            y_scale = lambda y: math.log10(y)
            x_inv = lambda x: 10**x
            y_inv = lambda y: 10**y
        else:
            x_scale = lambda x: x
            y_scale = lambda y: y
            x_inv = lambda x: x
            y_inv = lambda y: y
        
        x_min, x_max = x_scale(min_oi), x_scale(max_oi)
        y_min, y_max = y_scale(min_perf), y_scale(max_perf)
        
        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制坐标轴
        for i in range(height):
            canvas[i][0] = '│'
        for j in range(width):
            canvas[height-1][j] = '─'
        canvas[height-1][0] = '└'
        
        # 绘制计算上界 (水平线)
        compute_y = self._map_y(self.peak_gflops, y_min, y_max, height)
        if 0 <= compute_y < height:
            for j in range(1, width):
                canvas[compute_y][j] = '═'
            canvas[compute_y][0] = '├'
            # 标注
            label = f"Compute Peak: {self.peak_gflops:.0f} GFLOPS"
            for k, ch in enumerate(label):
                if compute_y > 0 and 1+k < width:
                    canvas[compute_y-1][1+k] = ch
        
        # 绘制内存上界 (斜线: Performance = BW * OI)
        for j in range(1, width):
            x = x_min + (x_max - x_min) * j / (width - 1)
            oi = x_inv(x)
            perf = self.memory_ceiling_gflops * oi
            y = self._map_y(perf, y_min, y_max, height)
            if 0 <= y < height:
                canvas[y][j] = '┈'
        
        # 绘制缓存上界
        for name, ceiling in self.cache_ceilings.items():
            for j in range(1, width):
                x = x_min + (x_max - x_min) * j / (width - 1)
                oi = x_inv(x)
                perf = ceiling * oi
                y = self._map_y(perf, y_min, y_max, height)
                if 0 <= y < height:
                    canvas[y][j] = '┄'
        
        # 绘制数据点
        symbols = {'compute': '●', 'memory': '■', 'latency': '▲'}
        for point in points:
            x = self._map_x(point.operational_intensity, x_min, x_max, width)
            y = self._map_y(point.achieved_gflops, y_min, y_max, height)
            if 0 <= x < width and 0 <= y < height:
                sym = symbols.get(point.bottleneck, '●')
                canvas[y][x] = sym
        
        # 转换为字符串
        lines = []
        for i in range(height):
            lines.append(''.join(canvas[i]).rstrip())
        
        # 添加图例
        lines.append("")
        lines.append("图例: ●=Compute-bound  ■=Memory-bound  ▲=Latency-bound")
        lines.append(f"Compute Peak: {self.peak_gflops:.0f} GFLOPS  |  DRAM BW: {self.memory_bw_gb_s:.0f} GB/s  |  Dtype: {self.dtype}")
        
        return '\n'.join(lines)
    
    def _map_x(self, val: float, x_min: float, x_max: float, width: int) -> int:
        # 防御: 通信等纯数据搬运算子 OI 可能为 0，log 域无定义，钳制到绘图下限
        val = max(val, 1e-2)
        x_norm = (math.log10(val) - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        return max(1, min(width - 2, int(x_norm * (width - 2))))
    
    def _map_y(self, val: float, y_min: float, y_max: float, height: int) -> int:
        val = max(val, 1e-2)
        y_norm = (math.log10(val) - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        return max(0, min(height - 1, height - 1 - int(y_norm * (height - 1))))
    
    def generate_plotly_data(self, result: RooflineAnalysisResult) -> Dict:
        """生成 Plotly 绘图数据"""
        
        # 背景上界线
        oi_range = [10**(i/10) for i in range(-20, 41)]  # 0.01 到 10000
        
        traces = []
        
        # 计算上界
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": f"Compute Peak ({self.peak_gflops:.0f} GFLOPS)",
            "x": oi_range,
            "y": [min(self.peak_gflops, self.memory_ceiling_gflops * oi) for oi in oi_range],
            "line": {"color": "red", "dash": "dash", "width": 2},
            "hoverinfo": "name",
        })
        
        # DRAM 内存上界
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": f"DRAM BW ({self.memory_bw_gb_s:.0f} GB/s)",
            "x": oi_range,
            "y": [self.memory_ceiling_gflops * oi for oi in oi_range],
            "line": {"color": "blue", "dash": "dot", "width": 2},
            "hoverinfo": "name",
        })
        
        # 缓存上界
        for name, ceiling in self.cache_ceilings.items():
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"{name} BW ({self.hardware.memory_levels[[m.name for m in self.hardware.memory_levels].index(name)].bandwidth_gb_s:.0f} GB/s)",
                "x": oi_range,
                "y": [ceiling * oi for oi in oi_range],
                "line": {"color": "green", "dash": "dashdot", "width": 1},
                "hoverinfo": "name",
            })
        
        # 数据点
        categories = {}
        for point in result.points:
            cat = point.category
            if cat not in categories:
                categories[cat] = {"x": [], "y": [], "text": [], "name": cat}
            categories[cat]["x"].append(point.operational_intensity)
            categories[cat]["y"].append(point.achieved_gflops)
            categories[cat]["text"].append(
                f"{point.name}<br>OI: {point.operational_intensity:.2f}<br>"
                f"Perf: {point.achieved_gflops:.1f} GFLOPS<br>"
                f"Bottleneck: {point.bottleneck}<br>Util: {point.utilization:.1%}"
            )
        
        colors = {
            "gemm": "purple", "attention": "orange", "activation": "cyan",
            "normalization": "magenta", "elementwise": "yellow", 
            "reduction": "brown", "memory": "gray", "communication": "pink"
        }
        
        for cat, data in categories.items():
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": cat,
                "x": data["x"],
                "y": data["y"],
                "text": data["text"],
                "marker": {"size": 12, "color": colors.get(cat, "black"), "symbol": "circle"},
                "hovertemplate": "%{text}<extra></extra>",
            })
        
        layout = {
            "title": f"NPU Roofline Model - {self.hardware.name} ({self.dtype.upper()})",
            "xaxis": {"type": "log", "title": "Operational Intensity (FLOPs/Byte)"},
            "yaxis": {"type": "log", "title": "Performance (GFLOPS)"},
            "hovermode": "closest",
            "showlegend": True,
            "width": 900,
            "height": 600,
        }
        
        return {"data": traces, "layout": layout}
    
    def generate_matplotlib_code(self, result: RooflineAnalysisResult) -> str:
        """生成可直接运行的 Matplotlib 绘图代码"""
        code = f'''
import matplotlib.pyplot as plt
import numpy as np

# 硬件参数
peak_gflops = {self.peak_gflops:.1f}
memory_bw_gb_s = {self.memory_bw_gb_s:.1f}
dtype = "{self.dtype}"

# 计算内存上界
bytes_per_op = {{"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}}.get(dtype, 2) * 2
memory_ceiling = memory_bw_gb_s * 1000 / bytes_per_op

# 算子数据点
operators = {{
'''
        for point in result.points:
            code += f'    "{point.name}": {{"oi": {point.operational_intensity:.4f}, "perf": {point.achieved_gflops:.2f}, "bottleneck": "{point.bottleneck}", "category": "{point.category}"}},\n'
        
        code += '''}

# 绘图
fig, ax = plt.subplots(figsize=(10, 7))
oi_range = np.logspace(-2, 4, 200)

# 计算上界
ax.plot(oi_range, np.minimum(peak_gflops, memory_ceiling * oi_range), 'r--', linewidth=2, label=f'Compute Peak ({peak_gflops:.0f} GFLOPS)')
# DRAM 上界
ax.plot(oi_range, memory_ceiling * oi_range, 'b:', linewidth=2, label=f'DRAM BW ({memory_bw_gb_s:.0f} GB/s)')

# 缓存上界
'''
        for name, ceiling in self.cache_ceilings.items():
            bw = next(m.bandwidth_gb_s for m in self.hardware.memory_levels if m.name == name)
            code += f'ax.plot(oi_range, {ceiling:.1f} * oi_range, "g-.", linewidth=1, label="{name} BW ({bw:.0f} GB/s)")\n'
        
        code += '''
# 数据点
colors = {"compute": "red", "memory": "blue", "latency": "orange"}
markers = {"gemm": "o", "attention": "s", "activation": "^", "normalization": "D", 
           "elementwise": "v", "reduction": "p", "memory": "X", "communication": "*"}

for name, data in operators.items():
    ax.scatter(data["oi"], data["perf"], 
               c=colors.get(data["bottleneck"], "black"),
               marker=markers.get(data["category"], "o"),
               s=100, alpha=0.8, label=name if name == list(operators.keys())[0] else "",
               edgecolors='white', linewidth=0.5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Operational Intensity (FLOPs/Byte)')
ax.set_ylabel('Performance (GFLOPS)')
ax.set_title(f'NPU Roofline Model - {self.hardware.name} ({dtype.upper()})')
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
'''
        return code
    
    def export_summary_table(self, result: RooflineAnalysisResult) -> str:
        """导出汇总表格 (Markdown 格式)"""
        lines = [
            f"# Roofline 分析报告: {self.hardware.name} ({self.dtype.upper()})",
            "",
            "## 硬件规格",
            f"- **峰值算力**: {self.peak_gflops:.0f} GFLOPS ({self.hardware.peak_tflops.get(self.dtype, 0):.1f} TFLOPS)",
            f"- **内存带宽**: {self.memory_bw_gb_s:.0f} GB/s",
            f"- **数据类型**: {self.dtype.upper()}",
            "",
            "## 算子分析汇总",
            "",
            "| 算子 | 类别 | 计算强度 (FLOPs/Byte) | 实际性能 (GFLOPS) | 峰值利用率 | 瓶颈类型 |",
            "|------|------|----------------------|-------------------|------------|----------|",
        ]
        
        for p in result.points:
            lines.append(
                f"| {p.name} | {p.category} | {p.operational_intensity:.2f} | "
                f"{p.achieved_gflops:.1f} | {p.utilization:.1%} | {p.bottleneck} |"
            )
        
        lines.extend([
            "",
            "## 整体统计",
            f"- **总算子数**: {result.summary['total_operators']}",
            f"- **总 FLOPs**: {result.summary['total_flops']:,}",
            f"- **总内存访问**: {result.summary['total_memory_bytes']:,} bytes",
            f"- **整体计算强度**: {result.summary['overall_arithmetic_intensity']:.2f} FLOPs/Byte",
            f"- **预估总耗时**: {result.summary['estimated_total_time_us']:.1f} μs",
            f"- **平均利用率**: {result.summary['average_utilization']:.1%}",
            f"- **计算受限算子**: {result.summary['compute_bound_ops']}",
            f"- **内存受限算子**: {result.summary['memory_bound_ops']}",
            f"- **延迟受限算子**: {result.summary['latency_bound_ops']}",
            "",
        ])
        
        if result.warnings:
            lines.append("## ⚠️ 优化建议")
            for w in result.warnings:
                lines.append(f"- {w}")
        
        return '\n'.join(lines)


__all__ = ["RooflineVisualizer"]