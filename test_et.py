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


# =========================
# Kernel 1: Prefix + ET + Compaction
# =========================
@triton.jit
def et_prefix_compact_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    GRID_N,
    K0, invK0, tau2, eps,
    TileIds_ptr, Counter_ptr, Acc_ptr, PassedMask_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_K0_TILES: tl.constexpr,
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
    s2  = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # prefix tiles
    K0_tiles = (K0 + BLOCK_K - 1) // BLOCK_K

    for kt in tl.static_range(0, MAX_K0_TILES):
        in_range = kt < K0_tiles
        k_base = kt * BLOCK_K
        k_ids = k_base + offs_k

        k_mask = in_range & (k_ids < DIN) & (k_ids < K0)

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

        contrib = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        acc += contrib
        s2  += contrib * contrib

    # squared test: acc^2 < tau^2 * (s2/K0 + eps)
    a2  = acc * acc
    rhs = tau2 * (s2 * invK0 + eps)
    passed = (a2 < rhs) & mask_mn

    cond_i32 = ((~passed) & mask_mn).to(tl.int32)
    row_sum = tl.sum(cond_i32, axis=1)      # [BM]
    active_cnt = tl.sum(row_sum, axis=0)    # scalar
    active_any = active_cnt > 0

    if not active_any:
        tl.store(
            Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            mask=mask_mn,
        )
        return

    # append survivor tile id
    slot = tl.atomic_add(Counter_ptr, 1)
    tile_id = pid_m * GRID_N + pid_n
    tl.store(TileIds_ptr + slot, tile_id)

    # store acc_prefix for survivor: Acc_ptr[slot, m, n]
    base_acc = Acc_ptr + slot * (BLOCK_M * BLOCK_N)
    idx_m = tl.arange(0, BLOCK_M)[:, None]
    idx_n = tl.arange(0, BLOCK_N)[None, :]
    tl.store(base_acc + idx_m * BLOCK_N + idx_n, acc, mask=mask_mn)

    # store passed mask per row as uint64 (BN<=64)
    # bit n=1 means passed for that (m,n)
    n_bits = tl.arange(0, BLOCK_N).to(tl.uint64)
    bit = (tl.full((BLOCK_N,), 1, tl.uint64) << n_bits)  # [BN]
    bit = bit * mask_n.to(tl.uint64)

    passed_u64 = passed.to(tl.uint64)  # [BM, BN]
    packed = tl.sum(passed_u64 * bit[None, :], axis=1)  # [BM] uint64

    base_pm = PassedMask_ptr + slot * BLOCK_M
    tl.store(base_pm + tl.arange(0, BLOCK_M), packed, mask=mask_m)


# =========================
# Kernel 2: Tail only on survivors
# =========================
@triton.jit
def et_tail_survivors_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    GRID_N,
    K0,
    TileIds_ptr, Acc_ptr, PassedMask_ptr,
    MAX_K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    sid = tl.program_id(0)

    tile_id = tl.load(TileIds_ptr + sid)
    # tile_id is scalar int; compute pid_m, pid_n
    pid_m = tile_id // GRID_N
    pid_n = tile_id - pid_m * GRID_N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < B
    mask_n = offs_n < DOUT
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # load prefix acc
    base_acc = Acc_ptr + sid * (BLOCK_M * BLOCK_N)
    idx_m = tl.arange(0, BLOCK_M)[:, None]
    idx_n = tl.arange(0, BLOCK_N)[None, :]
    acc = tl.load(base_acc + idx_m * BLOCK_N + idx_n, mask=mask_mn, other=0.0).to(tl.float32)

    # reconstruct passed from uint64 bitmask
    base_pm = PassedMask_ptr + sid * BLOCK_M
    packed = tl.load(base_pm + tl.arange(0, BLOCK_M), mask=mask_m, other=0).to(tl.uint64)

    n_bits = tl.arange(0, BLOCK_N).to(tl.uint64)
    bit = (tl.full((BLOCK_N,), 1, tl.uint64) << n_bits)
    passed = ((packed[:, None] & bit[None, :]) != 0) & mask_mn
    active_f = (~passed).to(tl.float32)

    start_tile = (K0 + BLOCK_K - 1) // BLOCK_K
    K_tiles = (DIN + BLOCK_K - 1) // BLOCK_K

    for kt in tl.static_range(0, MAX_K_TILES):
        in_range = (kt >= start_tile) & (kt < K_tiles)
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

        contrib = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        acc += contrib * active_f

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out = tl.where(passed, 0.0, acc)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=mask_mn,
    )


# =========================
# Two-kernel ET function
# =========================
def linear_et_two_kernel(
    x, w, b,
    K0=16,
    tau=-2.053748910631823,
    eps=1e-6,
    BM=32, BN=64, BK=32
):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    Bsz, DIN = x.shape
    DOUT = w.shape[0]
    assert BN <= 64, "This implementation packs passed mask into uint64; requires BN <= 64."

    y = torch.empty((Bsz, DOUT), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(Bsz, BM), triton.cdiv(DOUT, BN))
    GRID_N = grid[1]

    invK0 = 1.0 / float(K0)
    tau2 = float(tau) * float(tau)

    # Worst case: all tiles survive
    max_survivors = grid[0] * grid[1]
    tile_ids = torch.empty((max_survivors,), device=x.device, dtype=torch.int32)
    counter = torch.zeros((1,), device=x.device, dtype=torch.int32)

    acc_buf = torch.empty((max_survivors, BM, BN), device=x.device, dtype=torch.float32)
    passed_buf = torch.empty((max_survivors, BM), device=x.device, dtype=torch.uint64)

    MAX_K_TILES = triton.cdiv(DIN, BK)
    MAX_K0_TILES = triton.cdiv(K0, BK)

    # Kernel 1: prefix + enqueue survivors; stores zeros for all-pass tiles immediately
    et_prefix_compact_kernel[grid](
        x, w, b, y,
        Bsz, DIN, DOUT,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        GRID_N,
        K0, invK0, tau2, eps,
        tile_ids, counter, acc_buf, passed_buf,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        MAX_K0_TILES=MAX_K0_TILES
    )

    # NOTE: counter.item() synchronizes. For benchmarking kernel time, measure kernels separately or use CUDA graphs.
    n_survivors = int(counter.item())
    if n_survivors == 0:
        return y

    # Kernel 2: tail on survivors only
    et_tail_survivors_kernel[(n_survivors,)](
        x, w, b, y,
        Bsz, DIN, DOUT,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        GRID_N,
        K0,
        tile_ids, acc_buf, passed_buf,
        MAX_K_TILES=MAX_K_TILES,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    return y


# =========================
# main(): correctness + timing
# =========================
def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"

    device = "cuda"
    # Typical shapes — change these to match your layer
    B = 256
    DIN = 1024
    DOUT = 1024

    # Use fp16 inputs for tensor cores
    x = torch.randn(B, DIN, device=device, dtype=torch.float16)
    w = torch.randn(DOUT, DIN, device=device, dtype=torch.float16)  # same layout as nn.Linear.weight
    b = torch.randn(DOUT, device=device, dtype=torch.float16)


    # ET output
    y_et = linear_et_two_kernel(
        x, w, b,
        K0=16,
        tau=-2.053748910631823,
        eps=1e-6,
        BM=32, BN=64, BK=32,

    )

    print("y_et  shape:", tuple(y_et.shape), "dtype:", y_et.dtype)
    print("finite:", torch.isfinite(y_et).all().item())

    # Simple timing (includes the host sync for counter.item(); still useful as an end-to-end number)
    torch.cuda.synchronize()
    for _ in range(10):
        _ = linear_et_two_kernel(x, w, b, K0=16, tau=-2.053748910631823, BM=32, BN=64, BK=32)

    torch.cuda.synchronize()
    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = linear_et_two_kernel(x, w, b, K0=16, tau=-2.053748910631823, BM=32, BN=64, BK=32)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"ET two-kernel avg time: {ms:.3f} ms/iter")
    torch.cuda.synchronize()

    ms_ref = start.elapsed_time(end) / iters
    print(f"torch F.linear avg time: {ms_ref:.3f} ms/iter")


if __name__ == "__main__":
    main()
