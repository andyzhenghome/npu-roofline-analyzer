"""
模型解析器：从主流框架模型提取算子图
支持 PyTorch, ONNX, HuggingFace Transformers
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings

from .operators import OperatorInfo, OperatorCategory, registry
from .analyzer import NPURooflineAnalyzer


@dataclass
class ModelOp:
    """模型中的单个算子"""
    name: str
    op_type: str
    inputs: List[Dict]
    outputs: List[Dict]
    attrs: Dict[str, Any]
    source_location: str = ""  # 代码位置或层名


class ModelParser:
    """模型解析器基类"""
    
    def __init__(self, analyzer: NPURooflineAnalyzer):
        self.analyzer = analyzer
        self.ops: List[ModelOp] = []
    
    def parse(self, model, **kwargs) -> List[OperatorInfo]:
        """解析模型并返回算子分析列表"""
        self.ops = self._extract_ops(model, **kwargs)
        return self._analyze_ops()
    
    def _extract_ops(self, model, **kwargs) -> List[ModelOp]:
        raise NotImplementedError
    
    def _analyze_ops(self) -> List[OperatorInfo]:
        """将提取的算子转换为 OperatorInfo"""
        results = []
        for op in self.ops:
            op_info = self._convert_op(op)
            if op_info:
                results.append(op_info)
        return results
    
    def _convert_op(self, op: ModelOp) -> Optional[OperatorInfo]:
        """将通用算子转换为具体分析"""
        # 尝试从注册表获取分析器
        analyzer = registry.get_analyzer(op.op_type)
        if analyzer is None:
            warnings.warn(f"未知算子类型: {op.op_type}，跳过分析")
            return None
        
        # 根据算子类型提取参数
        try:
            kwargs = self._extract_op_params(op)
            return analyzer.analyze(**kwargs)
        except Exception as e:
            warnings.warn(f"分析算子 {op.name} ({op.op_type}) 失败: {e}")
            return None
    
    def _extract_op_params(self, op: ModelOp) -> Dict:
        """从算子属性提取分析参数 - 子类实现"""
        raise NotImplementedError


class PyTorchModelParser(ModelParser):
    """PyTorch 模型解析器 (基于 torch.fx)"""
    
    def _extract_ops(self, model, example_inputs=None, **kwargs) -> List[ModelOp]:
        try:
            import torch
            from torch.fx import symbolic_trace
        except ImportError:
            raise RuntimeError("需要安装 PyTorch: pip install torch")
        
        # 符号化追踪获取计算图
        if example_inputs is None:
            raise ValueError("PyTorch 解析需要提供 example_inputs")
        
        traced = symbolic_trace(model)
        graph = traced.graph
        
        ops = []
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method" or node.op == "call_module":
                # 获取模块名
                if node.op == "call_module":
                    mod_name = node.target
                    module = dict(model.named_modules()).get(mod_name, None)
                    op_type = type(module).__name__.lower() if module else "unknown"
                else:
                    op_type = node.target.__name__.lower() if hasattr(node.target, '__name__') else str(node.target)
                
                ops.append(ModelOp(
                    name=node.name,
                    op_type=op_type,
                    inputs=[{"name": str(arg)} for arg in node.args],
                    outputs=[{"name": node.name}],
                    attrs=dict(node.kwargs) if node.kwargs else {},
                    source_location=f"{node.op}:{node.target}"
                ))
        
        return ops
    
    def _extract_op_params(self, op: ModelOp) -> Dict:
        """从 PyTorch 节点提取参数 - 简化版，实际需结合 shape 信息"""
        # 这里需要结合 shape 传播，简化返回默认参数
        attrs = op.attrs
        return {
            "batch": attrs.get("batch", 1),
            "seq_len": attrs.get("seq_len", 2048),
            "hidden_dim": attrs.get("hidden_dim", 4096),
            "num_heads": attrs.get("num_heads", 32),
            "head_dim": attrs.get("head_dim", 128),
            "dtype": attrs.get("dtype", "fp16"),
        }


class ONNXModelParser(ModelParser):
    """ONNX 模型解析器"""
    
    def _extract_ops(self, model_path: str, **kwargs) -> List[ModelOp]:
        try:
            import onnx
        except ImportError:
            raise RuntimeError("需要安装 ONNX: pip install onnx")
        
        model = onnx.load(model_path)
        ops = []
        
        for node in model.graph.node:
            ops.append(ModelOp(
                name=node.name or node.output[0],
                op_type=node.op_type.lower(),
                inputs=[{"name": inp} for inp in node.input],
                outputs=[{"name": out} for out in node.output],
                attrs={attr.name: self._onnx_attr_to_python(attr) for attr in node.attribute},
                source_location=f"onnx:{node.op_type}"
            ))
        
        return ops
    
    def _onnx_attr_to_python(self, attr):
        import onnx
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode()
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        return None
    
    def _extract_op_params(self, op: ModelOp) -> Dict:
        attrs = op.attrs
        # ONNX 属性通常包含形状信息
        return {
            "batch": attrs.get("batch", 1),
            "seq_len": attrs.get("seq_len", attrs.get("sequence_length", 2048)),
            "hidden_dim": attrs.get("hidden_dim", attrs.get("hidden_size", 4096)),
            "num_heads": attrs.get("num_heads", attrs.get("num_attention_heads", 32)),
            "head_dim": attrs.get("head_dim", 128),
            "dtype": "fp16",  # ONNX 模型通常不含 dtype 信息
        }


class HFModelParser(ModelParser):
    """HuggingFace Transformers 模型解析器"""
    
    def _extract_ops(self, model, **kwargs) -> List[ModelOp]:
        """从 HF 模型提取标准 Transformer 结构"""
        config = model.config
        ops = []
        
        # 获取基本配置
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        intermediate_size = config.intermediate_size
        
        # 典型 Transformer Block 结构
        for layer_idx in range(num_layers):
            prefix = f"layer_{layer_idx}"
            
            # 1. Attention QKV 投影
            ops.append(ModelOp(f"{prefix}_qkv_proj", "gemm", [], [], {
                "batch": 1, "seq_len": 2048, "hidden_dim": hidden_size,
                "num_heads": num_heads, "head_dim": head_dim, "dtype": "fp16"
            }))
            
            # 2. Attention 核心
            ops.append(ModelOp(f"{prefix}_attention", "attention", [], [], {
                "batch": 1, "seq_len": 2048, "num_heads": num_heads,
                "head_dim": head_dim, "dtype": "fp16", "use_flash": True
            }))
            
            # 3. Attention Output 投影
            ops.append(ModelOp(f"{prefix}_attn_out", "gemm", [], [], {
                "M": 2048, "N": hidden_size, "K": hidden_size,
                "batch": 1, "dtype": "fp16"
            }))
            
            # 4. LayerNorm 1
            ops.append(ModelOp(f"{prefix}_ln1", "layernorm", [], [], {
                "batch": 1, "seq_len": 2048, "hidden_dim": hidden_size, "dtype": "fp16"
            }))
            
            # 5. FFN Up Proj
            ops.append(ModelOp(f"{prefix}_ffn_up", "gemm", [], [], {
                "M": 2048, "N": intermediate_size, "K": hidden_size,
                "batch": 1, "dtype": "fp16"
            }))
            
            # 6. Activation (SiLU/GELU)
            act_type = getattr(config, "hidden_act", "silu").lower()
            ops.append(ModelOp(f"{prefix}_act", act_type, [], [], {
                "num_elements": 2048 * intermediate_size, "dtype": "fp16"
            }))
            
            # 7. FFN Down Proj
            ops.append(ModelOp(f"{prefix}_ffn_down", "gemm", [], [], {
                "M": 2048, "N": hidden_size, "K": intermediate_size,
                "batch": 1, "dtype": "fp16"
            }))
            
            # 8. LayerNorm 2
            ops.append(ModelOp(f"{prefix}_ln2", "layernorm", [], [], {
                "batch": 1, "seq_len": 2048, "hidden_dim": hidden_size, "dtype": "fp16"
            }))
        
        return ops
    
    def _extract_op_params(self, op: ModelOp) -> Dict:
        return op.attrs  # HF parser 已在 attrs 中准备好参数


def create_parser(parser_type: str, analyzer: NPURooflineAnalyzer) -> ModelParser:
    """工厂函数创建解析器"""
    parsers = {
        "pytorch": PyTorchModelParser,
        "onnx": ONNXModelParser,
        "huggingface": HFModelParser,
        "hf": HFModelParser,
    }
    if parser_type.lower() not in parsers:
        raise ValueError(f"不支持的解析器类型: {parser_type}. 可选: {list(parsers.keys())}")
    return parsers[parser_type.lower()](analyzer)


__all__ = ["ModelParser", "PyTorchModelParser", "ONNXModelParser", "HFModelParser", "create_parser", "ModelOp"]