"""
Roofline 图渲染 (Pillow, 无 matplotlib 依赖)

混合视图: 同一模型在 H100 上的 prefill (B=1,L=2048) 与 decode (B=1,L=1,KV=2048) 算子
- 右上方聚集点: prefill，计算受限
- 对角线上聚集点: decode，内存受限
输出: roofline_mixed.png + demo_roofline_mixed.json
"""

import json
import math
import sys

sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

from npu_roofline_analyzer import NPUHardwareSpec, NPURooflineAnalyzer, registry

W, H = 1500, 960
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 120, 70, 80, 95

X_MIN, X_MAX = 0.1, 20000.0
Y_MIN, Y_MAX = 8000.0, 2_000_000.0

CATEGORY_COLORS = {
    "gemm": (123, 76, 214),
    "attention": (240, 140, 30),
    "activation": (0, 170, 190),
    "normalization": (210, 40, 150),
    "elementwise": (200, 165, 0),
    "reduction": (130, 90, 40),
    "memory": (110, 110, 110),
    "communication": (240, 80, 130),
}
CATEGORY_LABELS = {
    "gemm": "GEMM / Linear",
    "attention": "Attention",
    "activation": "Activation",
    "normalization": "Normalization",
    "elementwise": "Elementwise",
    "reduction": "Reduction",
    "memory": "Memory Op",
    "communication": "Communication",
}


def load_font(size, bold=False):
    names = [("msyhbd.ttc" if bold else "msyh.ttc"), ("arialbd.ttf" if bold else "arial.ttf")]
    for name in names:
        try:
            return ImageFont.truetype(rf"C:\Windows\Fonts\{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_ops():
    """构造 prefill + decode 两组算子"""
    B, L, Hd = 1, 2048, 4096
    heads, head_dim = 32, 128
    inter = 11008
    kv_len = 2048

    ops = []
    for layer in range(2):
        p = f"P{layer}"   # prefill
        ops.append(_named(registry.analyze("gemm", M=B * L, N=3 * Hd, K=Hd, batch=1, dtype="fp16"), f"{p}_QKV"))
        ops.append(_named(registry.analyze("attention", batch=B, seq_len=L, num_heads=heads,
                                           head_dim=head_dim, dtype="fp16", use_flash=True), f"{p}_Attn"))
        ops.append(_named(registry.analyze("gemm", M=B * L, N=Hd, K=Hd, batch=1, dtype="fp16"), f"{p}_AttnOut"))
        ops.append(_named(registry.analyze("layernorm", batch=B, seq_len=L, hidden_dim=Hd, dtype="fp16"), f"{p}_LN"))
        ops.append(_named(registry.analyze("gemm", M=B * L, N=inter, K=Hd, batch=1, dtype="fp16"), f"{p}_Gate"))
        ops.append(_named(registry.analyze("gemm", M=B * L, N=Hd, K=inter, batch=1, dtype="fp16"), f"{p}_Down"))
        ops.append(_named(registry.analyze("silu", num_elements=B * L * inter, dtype="fp16"), f"{p}_SiLU"))
        ops.append(_named(registry.analyze("mul", num_elements=B * L * inter, dtype="fp16"), f"{p}_Mul"))

        d = f"D{layer}"   # decode
        ops.append(_named(registry.analyze("gemm", M=1, N=3 * Hd, K=Hd, batch=1, dtype="fp16"), f"{d}_QKV"))
        ops.append(_named(registry.analyze("attention", batch=1, seq_len=1, seq_len_kv=kv_len, num_heads=heads,
                                           head_dim=head_dim, dtype="fp16", use_flash=True,
                                           kv_cache=True), f"{d}_Attn"))
        ops.append(_named(registry.analyze("gemm", M=1, N=Hd, K=Hd, batch=1, dtype="fp16"), f"{d}_AttnOut"))
        ops.append(_named(registry.analyze("layernorm", batch=1, seq_len=1, hidden_dim=Hd, dtype="fp16"), f"{d}_LN"))
        ops.append(_named(registry.analyze("gemm", M=1, N=inter, K=Hd, batch=1, dtype="fp16"), f"{d}_Gate"))
        ops.append(_named(registry.analyze("gemm", M=1, N=Hd, K=inter, batch=1, dtype="fp16"), f"{d}_Down"))
        ops.append(_named(registry.analyze("silu", num_elements=inter, dtype="fp16"), f"{d}_SiLU"))
        ops.append(_named(registry.analyze("add", num_elements=Hd, dtype="fp16", inplace=True), f"{d}_ResAdd"))
    return ops


def _named(op, name):
    op.name = name
    return op


def main():
    hw = NPUHardwareSpec.from_preset("h100")
    analyzer = NPURooflineAnalyzer(hw, dtype="fp16")
    result = analyzer.analyze_model(build_ops())
    analyzer.export_json(result, "demo_roofline_mixed.json")

    peak = analyzer.peak_gflops
    bw = hw.memory_bandwidth_gb_s
    bytes_per_op = 4  # fp16 读+写
    mem_k = bw * 1000 / bytes_per_op      # perf = mem_k * OI
    ridge_oi = peak / mem_k

    f_title = load_font(27, bold=True)
    f_sub = load_font(15)
    f_axis = load_font(18, bold=True)
    f_tick = load_font(15)
    f_small = load_font(14)
    f_leg = load_font(15)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # 坐标映射
    lx0 = math.log10(X_MIN)
    lx1 = math.log10(X_MAX)
    ly0 = math.log10(Y_MIN)
    ly1 = math.log10(Y_MAX)

    def px(x):
        x = max(x, X_MIN)
        return MARGIN_L + (math.log10(x) - lx0) / (lx1 - lx0) * (W - MARGIN_L - MARGIN_R)

    def py(y):
        y = max(y, Y_MIN)
        return (H - MARGIN_B) - (math.log10(y) - ly0) / (ly1 - ly0) * (H - MARGIN_T - MARGIN_B)

    # ---- 标题 ----
    d.text((MARGIN_L, 22), f"NPU Roofline — {hw.name} ({analyzer.dtype.upper()})", fill=(20, 20, 20), font=f_title)
    d.text((MARGIN_L, 56),
           f"peak {peak/1000:.0f} TFLOPS  ·  DRAM {bw:.0f} GB/s  ·  ridge OI = {ridge_oi:.2f}  ·  "
           f"operators {result.summary['total_operators']}  ·  avg util {result.summary['average_utilization']*100:.1f}%",
           fill=(95, 95, 95), font=f_sub)

    # ---- 网格 + 刻度 ----
    for xv in [0.1, 1, 10, 100, 1000, 10000]:
        X = px(xv)
        d.line([(X, MARGIN_T), (X, H - MARGIN_B)], fill=(234, 234, 234))
        d.text((X - 16, H - MARGIN_B + 11), f"{xv:g}", fill=(75, 75, 75), font=f_tick)
    for yv in [10000, 100000, 1000000]:
        Y = py(yv)
        d.line([(MARGIN_L, Y), (W - MARGIN_R, Y)], fill=(234, 234, 234))
        d.text((MARGIN_L - 62, Y - 9), f"{yv/1000:g}k", fill=(75, 75, 75), font=f_tick)

    # ---- 坐标轴 ----
    d.line([(MARGIN_L, MARGIN_T), (MARGIN_L, H - MARGIN_B)], fill=(55, 55, 55), width=2)
    d.line([(MARGIN_L, H - MARGIN_B), (W - MARGIN_R, H - MARGIN_B)], fill=(55, 55, 55), width=2)
    d.text(((MARGIN_L + W - MARGIN_R) // 2 - 140, H - MARGIN_B + 42),
           "Operational Intensity  (FLOPs / Byte)", fill=(25, 25, 25), font=f_axis)

    # Y 轴标题（旋转）
    ylabel = Image.new("RGBA", (400, 40), (255, 255, 255, 0))
    ImageDraw.Draw(ylabel).text((0, 0), "Performance (GFLOPS)", fill=(25, 25, 25), font=f_axis)
    ylabel = ylabel.rotate(90, expand=True)
    img.paste(ylabel, (30, (H - MARGIN_B + MARGIN_T) // 2 - 200), ylabel)

    # ---- Roofline 包络: min(compute, mem_k * OI) ----
    # 内存段 (脊点左侧)
    pts = []
    x = X_MIN
    while x <= ridge_oi * 1.02:
        pts.append((px(x), py(mem_k * x)))
        x *= 1.04
    pts.append((px(ridge_oi), py(peak)))
    d.line(pts, fill=(35, 85, 210), width=3)
    # 计算段 (脊点右侧)
    d.line([(px(ridge_oi), py(peak)), (px(X_MAX), py(peak))], fill=(215, 40, 40), width=3)

    # 脊点
    rx, ry = px(ridge_oi), py(peak)
    d.ellipse([rx - 7, ry - 7, rx + 7, ry + 7], fill=(120, 20, 20))
    d.text((rx + 13, ry + 12), f"ridge point  OI={ridge_oi:.2f}", fill=(120, 20, 20), font=f_small)

    # 上界线标注
    d.text((px(ridge_oi) + 40, py(peak) - 34), f"compute ceiling {peak:,.0f} GFLOPS",
           fill=(195, 35, 35), font=f_leg)
    d.text((px(0.16), py(mem_k * 0.16) + 22), f"memory ceiling {bw:.0f} GB/s",
           fill=(30, 65, 190), font=f_leg)

    # 区域文字
    d.text((px(X_MIN) + 30, MARGIN_T + 18), "memory-bound region", fill=(130, 130, 130), font=f_small)
    d.text((px(1200), MARGIN_T + 18), "compute-bound region", fill=(130, 130, 130), font=f_small)

    # ---- 数据点 ----
    drawn = {}
    for op in result.points:
        if op.achieved_gflops <= 0:
            continue
        oi = max(op.operational_intensity, 1e-3)
        x, y = px(oi), py(op.achieved_gflops)
        if not (MARGIN_L - 2 <= x <= W - MARGIN_R + 2 and MARGIN_T - 2 <= y <= H - MARGIN_B + 2):
            continue
        color = CATEGORY_COLORS.get(op.category, (90, 90, 90))
        r = 6 if op.category in ("gemm", "attention") else 4
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255))
        drawn[op.category] = drawn.get(op.category, 0) + 1

    # ---- 关键点标注 ----
    labels = {
        "P0_Attn": (9, -20),
        "P0_QKV": (-64, -22),
        "P0_SiLU": (10, 12),
        "P0_Mul": (9, -6),
        "D0_Attn": (-58, 10),
        "D0_Gate": (10, -18),
        "D0_QKV": (-58, 8),
    }
    for op in result.points:
        if op.name in labels and op.achieved_gflops > 0:
            dx, dy = labels[op.name]
            x, y = px(max(op.operational_intensity, 1e-3)), py(op.achieved_gflops)
            d.text((x + dx, y + dy), op.name, fill=(55, 55, 55), font=f_small)

    # ---- 图例 ----
    lx, ly = MARGIN_L + 18, MARGIN_T + 52
    for cat, color in CATEGORY_COLORS.items():
        if cat not in drawn:
            continue
        d.ellipse([lx, ly + 3, lx + 11, ly + 14], fill=color)
        d.text((lx + 18, ly), f"{CATEGORY_LABELS[cat]} ({drawn[cat]})", fill=(40, 40, 40), font=f_leg)
        ly += 25

    # ---- 底部说明 ----
    d.text((MARGIN_L, H - 20),
           "Points on/under the diagonal: bandwidth-limited.  Points under the horizontal line: compute-limited.",
           fill=(120, 120, 120), font=f_small)

    img.save("roofline_mixed.png")
    print("saved roofline_mixed.png")
    print("drawn:", sum(drawn.values()), drawn)
    print(f"ridge OI={ridge_oi:.2f}, mem_k={mem_k:.0f} GFLOPS/(FLOP/Byte)")
    for p in result.points:
        if p.name.startswith("D0") or p.name.startswith("P0"):
            print(f"  {p.name:12s} OI={p.operational_intensity:8.2f}  perf={p.achieved_gflops:10.0f}  {p.bottleneck}")


if __name__ == "__main__":
    main()