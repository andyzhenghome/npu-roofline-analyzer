#!/usr/bin/env python3
"""
NPU Roofline 静态分析工具 - CLI 入口
"""

import argparse
import sys
import json
from pathlib import Path

from npu_roofline_analyzer import (
    NPUHardwareSpec,
    NPURooflineAnalyzer,
    RooflineVisualizer,
    registry,
    create_parser,
)


def cmd_list_operators(args):
    """列出所有支持的算子"""
    ops = registry.list_operators()
    print("支持的算子:")
    for op in sorted(ops):
        analyzer = registry.get_analyzer(op)
        cat = analyzer.get_category().value if analyzer else "unknown"
        print(f"  {op:30s} [{cat}]")


def cmd_list_hardware(args):
    """列出预设硬件配置"""
    presets = ["h100", "a100", "ascend910b", "generic_npu"]
    print("预设硬件配置:")
    for p in presets:
        spec = NPUHardwareSpec.from_preset(p)
        print(f"  {p:15s} - {spec.name}")
        print(f"    FP16: {spec.peak_tflops.get('fp16', 0)} TFLOPS, "
              f"BW: {spec.memory_bandwidth_gb_s} GB/s")


def cmd_analyze_operator(args):
    """分析单个算子"""
    hardware = NPUHardwareSpec.from_preset(args.hardware)
    analyzer = NPURooflineAnalyzer(hardware, dtype=args.dtype)
    visualizer = RooflineVisualizer(hardware, dtype=args.dtype)
    
    # 解析参数
    kwargs = {}
    if args.params:
        for param in args.params:
            if '=' in param:
                k, v = param.split('=', 1)
                # 尝试转换类型
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        if v.lower() in ('true', 'false'):
                            v = v.lower() == 'true'
                kwargs[k] = v
    
    point = analyzer.analyze_by_name(args.operator, **kwargs)
    if point is None:
        print(f"错误: 未知算子 '{args.operator}' 或参数不足")
        print("使用 --list-ops 查看支持的算子及参数")
        sys.exit(1)
    
    # 输出结果
    if args.format == "json":
        print(json.dumps({
            "name": point.name,
            "category": point.category,
            "operational_intensity": point.operational_intensity,
            "achieved_gflops": point.achieved_gflops,
            "peak_gflops": point.peak_gflops,
            "memory_bandwidth_gb_s": point.memory_bandwidth_gb_s,
            "compute_ceiling_gflops": point.compute_ceiling,
            "memory_ceiling_gflops": point.memory_ceiling,
            "bottleneck": point.bottleneck,
            "utilization": point.utilization,
        }, indent=2))
    else:
        print(f"算子: {point.name} ({point.category})")
        print(f"  计算强度: {point.operational_intensity:.2f} FLOPs/Byte")
        print(f"  实际性能: {point.achieved_gflops:.1f} GFLOPS")
        print(f"  峰值性能: {point.peak_gflops:.1f} GFLOPS")
        print(f"  利用率:   {point.utilization:.1%}")
        print(f"  瓶颈:     {point.bottleneck}")
        print(f"  计算上界: {point.compute_ceiling:.1f} GFLOPS")
        print(f"  内存上界: {point.memory_ceiling:.1f} GFLOPS (@{point.memory_bandwidth_gb_s:.0f} GB/s)")


def cmd_analyze_model(args):
    """分析完整模型"""
    hardware = NPUHardwareSpec.from_preset(args.hardware)
    analyzer = NPURooflineAnalyzer(hardware, dtype=args.dtype)
    visualizer = RooflineVisualizer(hardware, dtype=args.dtype)
    
    # 解析模型
    if args.model_type == "transformer":
        # 使用内置 Transformer 示例
        from examples.transformer_analysis import analyze_llama_like_model
        result = analyze_llama_like_model()
    elif args.model_type == "inference":
        from examples.transformer_analysis import analyze_inference_kv_cache
        result = analyze_inference_kv_cache()
    elif args.model_type == "moe":
        from examples.transformer_analysis import analyze_moe_model
        result = analyze_moe_model()
    elif args.model_type == "pytorch":
        if not args.model_path:
            print("错误: PyTorch 模型需要 --model-path")
            sys.exit(1)
        parser = create_parser("pytorch", analyzer)
        # 需要用户提供模型加载代码
        print("PyTorch 模型解析需要自定义加载逻辑，请参考文档")
        sys.exit(1)
    elif args.model_type == "onnx":
        if not args.model_path:
            print("错误: ONNX 模型需要 --model-path")
            sys.exit(1)
        parser = create_parser("onnx", analyzer)
        result = parser.parse(args.model_path)
    elif args.model_type == "huggingface":
        if not args.model_path:
            print("错误: HF 模型需要 --model-path")
            sys.exit(1)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        parser = create_parser("huggingface", analyzer)
        result = parser.parse(model)
    else:
        print(f"未知模型类型: {args.model_type}")
        sys.exit(1)
    
    # 输出
    if args.output:
        analyzer.export_json(result, args.output + ".json")
        analyzer.export_csv(result, args.output + ".csv")
        print(f"结果已导出: {args.output}.json, {args.output}.csv")
    
    if args.plot:
        plotly_data = visualizer.generate_plotly_data(result)
        with open(args.plot + "_plotly.json", "w") as f:
            json.dump(plotly_data, f, indent=2)
        
        mpl_code = visualizer.generate_matplotlib_code(result)
        with open(args.plot + "_mpl.py", "w") as f:
            f.write(mpl_code)
        print(f"绘图文件已生成: {args.plot}_plotly.json, {args.plot}_mpl.py")
    
    if not args.quiet:
        print(visualizer.generate_ascii_plot(result, width=100, height=30))
        print()
        print(visualizer.export_summary_table(result))


def cmd_custom_hardware(args):
    """自定义硬件配置"""
    spec = NPUHardwareSpec(
        name=args.name,
        peak_tflops=json.loads(args.peak_tflops),
        memory_bandwidth_gb_s=args.bandwidth,
        compute_units=json.loads(args.compute_units),
        clock_ghz=args.clock,
    )
    print(f"自定义硬件: {spec.name}")
    print(f"  峰值算力: {spec.peak_tflops}")
    print(f"  内存带宽: {spec.memory_bandwidth_gb_s} GB/s")
    print(f"  计算单元: {spec.compute_units}")
    print(f"  频率: {spec.clock_ghz} GHz")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "name": spec.name,
                "peak_tflops": spec.peak_tflops,
                "memory_bandwidth_gb_s": spec.memory_bandwidth_gb_s,
                "compute_units": spec.compute_units,
                "clock_ghz": spec.clock_ghz,
            }, f, indent=2)
        print(f"配置已保存到: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="NPU Roofline 静态分析工具 - 用于大模型算子性能建模",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出支持的算子
  npu-roofline list-ops
  
  # 列出预设硬件
  npu-roofline list-hardware
  
  # 分析单个 GEMM 算子
  npu-roofline analyze-op gemm --hardware h100 --dtype fp16 \\
      --params M=4096 N=4096 K=4096 batch=1
  
  # 分析 Attention 算子
  npu-roofline analyze-op attention --hardware h100 \\
      --params batch=1 seq_len=2048 num_heads=32 head_dim=128 use_flash=true
  
  # 分析完整 Transformer 模型
  npu-roofline analyze-model transformer --hardware h100 \\
      --output llama_result --plot llama_plot
  
  # 分析推理阶段
  npu-roofline analyze-model inference --hardware h100
  
  # 分析 MoE 模型
  npu-roofline analyze-model moe --hardware h100
  
  # 自定义硬件配置
  npu-roofline custom-hardware --name "MyNPU" \\
      --peak-tflops '{"fp16": 200, "fp32": 50}' \\
      --bandwidth 800 --compute-units '{"mac": 2048}' --clock 1.2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # list-ops
    subparsers.add_parser("list-ops", help="列出所有支持的算子")
    
    # list-hardware
    subparsers.add_parser("list-hardware", help="列出预设硬件配置")
    
    # analyze-op
    op_parser = subparsers.add_parser("analyze-op", help="分析单个算子")
    op_parser.add_argument("operator", help="算子名称 (如 gemm, attention, layernorm)")
    op_parser.add_argument("--hardware", default="h100", help="硬件预设")
    op_parser.add_argument("--dtype", default="fp16", help="数据类型")
    op_parser.add_argument("--params", nargs="+", help="算子参数 key=value 形式")
    op_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    # analyze-model
    model_parser = subparsers.add_parser("analyze-model", help="分析完整模型")
    model_parser.add_argument("model_type", choices=["transformer", "inference", "moe", "pytorch", "onnx", "huggingface"])
    model_parser.add_argument("--hardware", default="h100", help="硬件预设")
    model_parser.add_argument("--dtype", default="fp16", help="数据类型")
    model_parser.add_argument("--model-path", help="模型文件路径 (onnx/hf/pytorch)")
    model_parser.add_argument("--output", help="输出文件前缀")
    model_parser.add_argument("--plot", help="绘图输出前缀")
    model_parser.add_argument("--quiet", action="store_true", help="不打印 ASCII 图表")
    
    # custom-hardware
    hw_parser = subparsers.add_parser("custom-hardware", help="创建自定义硬件配置")
    hw_parser.add_argument("--name", required=True, help="硬件名称")
    hw_parser.add_argument("--peak-tflops", required=True, help="JSON 格式峰值算力，如 '{\"fp16\": 200}'")
    hw_parser.add_argument("--bandwidth", type=float, required=True, help="内存带宽 GB/s")
    hw_parser.add_argument("--compute-units", required=True, help="JSON 格式计算单元，如 '{\"mac\": 2048}'")
    hw_parser.add_argument("--clock", type=float, default=1.0, help="核心频率 GHz")
    hw_parser.add_argument("--output", help="保存配置到文件")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        "list-ops": cmd_list_operators,
        "list-hardware": cmd_list_hardware,
        "analyze-op": cmd_analyze_operator,
        "analyze-model": cmd_analyze_model,
        "custom-hardware": cmd_custom_hardware,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()