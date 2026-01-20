import os
import torch, math
import triton
import triton.language as tl
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from scipy.stats import norm

import triton
import triton.language as tl


import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_baseline_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    MAX_K_TILES: tl.constexpr,      # ceil_div(DIN, BK)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < B
    mask_n = offs_n < DOUT
    mask_mn = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_tiles = (DIN + BLOCK_K - 1) // BLOCK_K

    for kt in tl.static_range(0, MAX_K_TILES):
        in_range = kt < K_tiles
        k_base = kt * BLOCK_K
        k_ids = k_base + offs_k
        k_mask = in_range & (k_ids < DIN)

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=mask_mn,
    )


def linear_baseline_triton(x, w, b, BM=32, BN=64, BK=32):
    Bsz, DIN = x.shape
    DOUT = w.shape[0]
    y = torch.empty((Bsz, DOUT), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(Bsz, BM), triton.cdiv(DOUT, BN))
    MAX_K_TILES = triton.cdiv(DIN, BK)

    linear_baseline_kernel[grid](
        x, w, b, y,
        Bsz, DIN, DOUT,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        MAX_K_TILES=MAX_K_TILES,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return y

def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    B = 256
    DIN = 1024
    DOUT = 1024

    x = torch.randn(B, DIN, device=device, dtype=torch.float16)
    w = torch.randn(DOUT, DIN, device=device, dtype=torch.float16)
    b = torch.randn(DOUT, device=device, dtype=torch.float16)

    BM, BN, BK = 32, 64, 32

    y_base = linear_baseline_triton(x, w, b, BM=BM, BN=BN, BK=BK)

    print("baseline triton:", y_base.shape, y_base.dtype, "finite:", torch.isfinite(y_base).all().item())
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()