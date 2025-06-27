import torch
from typing import Tuple

# -------- INT8 量化 --------
def quantize_per_tensor_absmax(x: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, float]:
    q_max = 2 ** (n_bits - 1) - 1
    max_val = x.abs().max()
    scale = (max_val / q_max).clamp(min=1e-5)
    q_x = (x / scale).clamp(-q_max, q_max).round()
    return q_x, scale

def dequantize_int(q_x: torch.Tensor, scale: float) -> torch.Tensor:
    return q_x * scale

# -------- FP8 量化 --------
def per_tensor_quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        amax = torch.tensor(16.0, dtype=tensor.dtype)
    else:
        min_val, max_val = tensor.aminmax()
        amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    q_w = (tensor * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    inv_scale = scale.float().reciprocal().item()
    return q_w, inv_scale

def dequantize_fp8(q_x: torch.Tensor, inv_scale: float) -> torch.Tensor:
    return q_x.to(torch.float32) * inv_scale

# -------- 测试脚本 --------
W = torch.rand(2, 3, dtype=torch.float32)
X = torch.rand(3, 4, dtype=torch.float32)
Y = W @ X

print("=== 原始矩阵 ===")
print("W:\n", W)
print("X:\n", X)
print("Y = W @ X:\n", Y)

# INT8 测试
W_q8, W_s8 = quantize_per_tensor_absmax(W)
X_q8, X_s8 = quantize_per_tensor_absmax(X)
Y_q8 = W_q8 @ X_q8
Y_dq8 = dequantize_int(Y_q8, W_s8 * X_s8)

print("\n=== INT8 量化测试 ===")
print("W_q8:\n", W_q8, "\nScale:", W_s8)
print("X_q8:\n", X_q8, "\nScale:", X_s8)
print("Y_q8 = W_q8 @ X_q8:\n", Y_q8) # 大于int8范围，累加值需要int32来保存
print("Y_dq8 ≈ Y:\n", Y_dq8)

# FP8 测试
W_fp8, W_inv_fp8 = per_tensor_quantize_fp8(W)
X_fp8, X_inv_fp8 = per_tensor_quantize_fp8(X)
W_dq_fp8 = dequantize_fp8(W_fp8, W_inv_fp8)
X_dq_fp8 = dequantize_fp8(X_fp8, X_inv_fp8)
Y_dq_fp8 = W_dq_fp8 @ X_dq_fp8

print("\n=== FP8 量化测试 ===")
print("W_fp8:\n", W_fp8, "\nInvScale:", W_inv_fp8)
print("X_fp8:\n", X_fp8, "\nInvScale:", X_inv_fp8)
print("W_dq_fp8 ≈ W:\n", W_dq_fp8)
print("X_dq_fp8 ≈ X:\n", X_dq_fp8)
print("Y_dq_fp8 ≈ Y:\n", Y_dq_fp8)

# # https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/quantize.py