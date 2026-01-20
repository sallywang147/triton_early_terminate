import os
import torch, math
import triton
import triton.language as tl
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from scipy.stats import norm


# =============================
# Python Module Wrapper
# =============================

 # two-tailed 95% confidence

@triton.jit
def linear_etu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT, 
     stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
          # early-termination prefix length   # max tail iterations (compile-time) 
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    pid_m = tl.program_id(0)   # batch row
    pid_n = tl.program_id(1)   # output col block

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < B
    mask_n = offs_n < 64

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_mask = offs_n < DOUT
    y1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    s2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    K0 = 16

    for k0 in range(0, K0, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Guard within [0, K0) and also within [0, K)
        k_mask = (k_ids < K0) & (k_ids < BLOCK_K)

        x_vals = tl.load(
            X_ptr + pid_m * stride_xm + k_ids * stride_xk,
            mask=k_mask,
            other=0.0,
        )

        w_vals = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        prod = w_vals * x_vals[None, :]
        tmp = tl.sum(prod, axis=1)          # contribution to y1
        y1 += tmp
        s2 += tl.sum(prod * prod, axis=1)   # contribution to s2
    tau = -2.053748910631823
    denom = tl.sqrt(s2 / K0 )
    tstat = tl.abs(y1) / denom
    passed = (tstat < tau) & n_mask        # output lanes that pass ET
    acc = y1  
    still_active = (~passed) & n_mask
    for k0 in range(K0, DOUT, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < DOUT

        # If nothing is active, this will be masked out anyway
        w_vals = tl.load(
            W_ptr + k_ids[None, :] * stride_wk,
            mask=still_active[:, None] & k_mask[None, :],
            other=0.0,
        )                                       # [BLOCK_N, BLOCK_K]

        any_active = tl.sum(still_active) > 0
        x_vals = tl.load(
            X_ptr + k_ids * stride_xk,
            mask=k_mask & any_active,
            other=0.0,
        )                                       # [BLOCK_K]

        prod = w_vals * x_vals[None, :]         # [BLOCK_N, BLOCK_K]
        partial = tl.sum(prod, axis=1)          # [BLOCK_N]
        acc += partial * still_active.to(tl.float32)

    # Add bias
    out = tl.where(passed, 0.0, acc)
    tl.store(
        Y_ptr + pid_m * stride_ym + offs_n * stride_yn,
        out,
        mask=n_mask,
    )

    # --------------------------
    # Phase 2: tail [K0, DIN)
    # --------------------------
    for k in range(K0, DIN, BLOCK_K):
        k_ids = k + offs_k
        k_mask = k_ids < DIN

        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )

        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        )

        contrib = tl.dot(x, tl.trans(w))
        active = (~passed).to(tl.float32)
        acc += contrib * active

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Final output: zero out passed lanes (like your MyLinear)
    out = tl.where(passed, 0.0, acc)

    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=mask_mn,
    )


class TritonLinearET(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,

        BLOCK_M: int = 32,
        BLOCK_N: int = 64,
        BLOCK_K: int = 32,
    ):
        super().__init__()
        self.weight = linear.weight.detach().contiguous()
        self.bias = linear.bias.detach().contiguous()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.BM = BLOCK_M
        self.BN = BLOCK_N
        self.BK = BLOCK_K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, DIN = x.shape
        assert DIN == self.in_features

        DOUT = self.out_features
        y = torch.empty((B, DOUT), device=x.device, dtype=torch.float32)

        stride_xm, stride_xk = x.stride()
        stride_wn, stride_wk = self.weight.stride()
        stride_ym, stride_yn = y.stride()

        grid = (
            triton.cdiv(B, self.BM),
            triton.cdiv(DOUT, self.BN),
        )
    
        linear_etu_kernel[grid](
            x, self.weight, self.bias, y,
            B, DIN, DOUT,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M = self.BM,
            BLOCK_N=self.BN,
            BLOCK_K= self.BK,
  
        )

        return y




# =============================
# Test
# =============================

def main():
    torch.manual_seed(0)
    layer = nn.Linear(256, 1024).cuda().float()
    tri_et = TritonLinearET(layer).cuda()
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    _ = tri_et(x)
    torch.cuda.synchronize()
    


if __name__ == "__main__":
    main()