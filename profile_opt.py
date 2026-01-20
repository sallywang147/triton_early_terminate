-import os
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.cuda.nvtx as nvtx
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# Baseline Triton Linear Kernel
# =============================

@triton.jit
def triton_linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # row block
    pid_n = tl.program_id(1)   # col block

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < B
    mask_n = offs_n < DOUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, DIN, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < DIN

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]

        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ==================================
# Early-Termination Triton Linear (1D)
# ==================================
@triton.jit
def linear_etu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Simple ET toy kernel:
      - Prefix on K0=16, t-stat test, then masked tail.
      - Works on X: [B, DIN], W: [DOUT, DIN], Y: [B, DOUT].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < B
    mask_n = offs_n < DOUT
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # prefix length and tau are just constants for now
    K0 = 64
    tau = -2.053748910631823
    eps = 0

    # ---- Phase 1: prefix [0, K0) ----
    y1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    s2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K0, BLOCK_K):
        k_ids = k0 + offs_k
        k_mask = (k_ids < DIN) & (k_ids < K0)

        x_vals = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]

        w_vals = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        )  # [BLOCK_N, BLOCK_K]

        prod = tl.dot(x_vals, tl.trans(w_vals))  # [BLOCK_M, BLOCK_N]
        y1 += prod
        s2 += prod * prod

    denom = tl.sqrt(s2 / K0 + eps)
    tstat = tl.abs(y1) / denom
    passed = (tstat < tau) & mask_mn
    acc = y1

    # ---- Phase 2: tail [K0, DIN) ----
    for k0 in range(K0, DIN, BLOCK_K):
        k_ids = k0 + offs_k
        k_mask = k_ids < DIN

        x_vals = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_vals = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        )

        prod = tl.dot(x_vals, tl.trans(w_vals))
        active = (~passed).to(tl.float32)
        acc += prod * active

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    out = tl.where(passed, 0.0, acc)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=mask_mn,
    )




# =============================
# Python Wrappers (tri, tri_et)
# =============================

class TritonLinear(nn.Module):
    """
    tri: baseline Triton matmul, drop-in for nn.Linear.
    Supports inputs of shape [..., in_features].
    """

    def __init__(self, linear, BLOCK_M=32, BLOCK_N=64, BLOCK_K=32):
        super().__init__()
        # For inference-only profiling we keep weights as plain tensors
        self.weight = linear.weight.detach().contiguous()
        self.bias = (
            linear.bias.detach().contiguous()
            if linear.bias is not None
            else None
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.BM = BLOCK_M
        self.BN = BLOCK_N
        self.BK = BLOCK_K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        DIN = orig_shape[-1]
        assert DIN == self.in_features, f"Expected in_features={self.in_features}, got {DIN}"

        # Flatten leading dims to 2D
        x_2d = x.reshape(-1, DIN).contiguous()
        B = x_2d.shape[0]
        DOUT = self.out_features

        y_2d = torch.empty((B, DOUT), device=x.device, dtype=torch.float32)

        grid = (
            triton.cdiv(B, self.BM),
            triton.cdiv(DOUT, self.BN),
        )

        triton_linear_kernel[grid](
            x_2d, self.weight, self.bias, y_2d,
            B, DIN, DOUT,
            x_2d.stride(0), x_2d.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y_2d.stride(0), y_2d.stride(1),
            BLOCK_M=self.BM,
            BLOCK_N=self.BN,
            BLOCK_K=self.BK,
        )

        # Reshape back
        y = y_2d.reshape(*orig_shape[:-1], DOUT)
        # Cast back to original dtype if needed
        return y.to(x.dtype)


class TritonLinearET(nn.Module):
    """
    tri_early: early-termination Triton matmul, drop-in for nn.Linear.
    Supports inputs of shape [..., in_features].
    """

    def __init__(
        self,
        linear: nn.Linear,
        BLOCK_M: int = 64,
        BLOCK_N: int = 128,
        BLOCK_K: int = 64,
    ):
        super().__init__()
        self.weight = linear.weight.detach().contiguous()
        self.bias = (
            linear.bias.detach().contiguous()
            if linear.bias is not None
            else None
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.BM = BLOCK_M
        self.BN = BLOCK_N
        self.BK = BLOCK_K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        DIN = orig_shape[-1]
        assert DIN == self.in_features, f"Expected in_features={self.in_features}, got {DIN}"

        x_2d = x.reshape(-1, DIN).contiguous()
        B = x_2d.shape[0]
        DOUT = self.out_features

        y_2d = torch.empty((B, DOUT), device=x.device, dtype=torch.float32)

        grid = (
            triton.cdiv(B, self.BM),
            triton.cdiv(DOUT, self.BN),
        )

        linear_etu_kernel[grid](
            x_2d, self.weight, self.bias, y_2d,
            B, DIN, DOUT,
            x_2d.stride(0), x_2d.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y_2d.stride(0), y_2d.stride(1),
            BLOCK_M=self.BM,
            BLOCK_N=self.BN,
            BLOCK_K=self.BK,
        )

        y = y_2d.reshape(*orig_shape[:-1], DOUT)
        return y.to(x.dtype)


# =======================================
# Patch OPT-125M: Q/K/V → tri, fc1 → tri_et
# =======================================

def patch_opt_with_triton(model: nn.Module):
    """
    For OPT-125M:
      - self_attn.q_proj, k_proj, v_proj → TritonLinear (tri)
      - fc1 → TritonLinearET (tri_early)
      - everything else stays as nn.Linear
    """
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        nvtx.range_push("OPT_Triton_QKV_FC1")
        # Q, K, V projections: baseline Triton
        attn.q_proj = TritonLinearET(attn.q_proj)
        attn.k_proj = TritonLinearET(attn.k_proj)
        attn.v_proj = TritonLinearET(attn.v_proj)
        # fc1: early termination Triton
        layer.fc1 = TritonLinearET(layer.fc1)
        torch.cuda.synchronize()
        nvtx.range_pop()
    print("✅ Patched OPT: q_proj/k_proj/v_proj → TritonLinear, fc1 → TritonLinearET")


def patch_opt_with_Linear(model, use_triton=True, use_et=False):
    # Q, K, V: Triton or ET Triton
    for layer in model.model.decoder.layers:
        attn = layer.self_attn

        # -------- Q PROJ ------------
        old = attn.q_proj
        if use_triton and not use_et:
            attn.q_proj = TritonLinear(old).to(old.weight.device)
        elif use_et:
            attn.q_proj = TritonLinearET(old).to(old.weight.device)
        else:
            attn.q_proj = nn.Linear(old.in_features, old.out_features).to(old.weight.device)

        # -------- K PROJ ------------
        old = attn.k_proj
        if use_triton and not use_et:
            attn.k_proj = TritonLinear(old).to(old.weight.device)
        elif use_et:
            attn.k_proj = TritonLinearET(old).to(old.weight.device)
        else:
            attn.k_proj = nn.Linear(old.in_features, old.out_features).to(old.weight.device)

        # -------- V PROJ ------------
        old = attn.v_proj
        if use_triton and not use_et:
            attn.v_proj = TritonLinear(old).to(old.weight.device)
        elif use_et:
            attn.v_proj = TritonLinearET(old).to(old.weight.device)
        else:
            attn.v_proj = nn.Linear(old.in_features, old.out_features).to(old.weight.device)

        # -------- FC1 ------------
        old = layer.fc1
        if use_triton and not use_et:
            layer.fc1 = TritonLinear(old).to(old.weight.device)
        elif use_et:
            layer.fc1 = TritonLinearET(old).to(old.weight.device)
        else:
            layer.fc1 = nn.Linear(old.in_features, old.out_features).to(old.weight.device)
# =============================
# Tiny sanity test (single layer)
# =============================

def _tiny_sanity_test():
    torch.manual_seed(0)
    layer = nn.Linear(32, 64).cuda().float()
    tri = TritonLinear(layer).cuda()
    tri_et = TritonLinearET(layer).cuda()

    x = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    y_ref = layer(x)
    y_tri = tri(x)
    y_early = tri_et(x)

    print("ref vs tri max diff:", (y_ref - y_tri).abs().max().item())
    print("ref vs tri_et (not expected equal due to ET logic) max diff:",
          (y_ref - y_early).abs().max().item())


# =============================
# Example: load OPT + patch
# =============================

def main():
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float32,   # keep things simple for Triton
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Patch Q/K/V + fc1
    patch_opt_with_triton(model)
    #patch_opt_with_Linear(model)

    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    print("Logits mean:", out.logits.float().mean().item())


if __name__ == "__main__":
    # Optional: quick unit test on a toy layer
    #_tiny_sanity_test()
    # Then run the real OPT test
    main()
