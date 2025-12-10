import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from scipy.stats import norm
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.cuda.nvtx as nvtx


# ============================================================
# Triton Kernel: Early-Termination Linear
# ============================================================

@triton.jit
def linear_baseline_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, N, K,
    stride_xb, stride_xk,
    stride_wn, stride_wk,
    stride_yb, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Baseline matmul: y = x @ W^T
    - X: [B, K]
    - W: [N, K]
    - Y: [B, N]
    Each program handles one row b and BLOCK_N output channels.
    No early termination, no stats, just full matmul.
    """
    b = tl.program_id(0)          # row index
    n_block = tl.program_id(1)    # block index over N

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # initialize accumulator for BLOCK_N outputs
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < K

        # load X row slice: shape [BLOCK_K]
        x_vals = tl.load(
            X_ptr + b * stride_xb + k_ids * stride_xk,
            mask=k_mask,
            other=0.0,
        )

        # load W slice: shape [BLOCK_N, BLOCK_K]
        w_vals = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # accumulate dot product for each output lane n
        prod = w_vals * x_vals[None, :]
        acc += tl.sum(prod, axis=1)

    # store result
    tl.store(
        Y_ptr + b * stride_yb + offs_n * stride_yn,
        acc,
        mask=n_mask,
    )


@triton.jit
def linear_etu_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, N, K,
    tau, eps,
    stride_xb, stride_xk,
    stride_wn, stride_wk,
    stride_yb, stride_yn,
    K0: tl.constexpr,          # early-termination prefix length
    MAX_ITERS: tl.constexpr,   # max tail iterations (compile-time)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes y = x @ W^T with early termination on output lanes.

    - X: (B, K)
    - W: (N, K)  [out, in]
    - Y: (B, N)

    Phase 1: compute first K0 columns:
        y1 = sum_{k < K0} x_k * w_k
        s2 = sum_{k < K0} (x_k * w_k)^2

    Confidence test per output element:
        passed = |y1| / sqrt( s2 / K0 + eps ) > tau

    Phase 2: for outputs that did NOT pass, accumulate remaining columns
        k ∈ [K0, K)

    Implementation constraints:
      - No dynamic break/continue; loops are static in terms of K0, MAX_ITERS.
      - Early termination is implemented via boolean masks on lanes.
    """
    # program ids
    b = tl.program_id(0)               # row of X
    n_block = tl.program_id(1)         # block across N

    # N offsets for this block
    n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # ------------------------------------------------
    # Phase 1: first K0 columns
    # ------------------------------------------------
    y1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    s2 = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K0, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Guard within [0, K0) and also within [0, K)
        k_mask = (k_ids < K0) & (k_ids < K)

        x_vals = tl.load(
            X_ptr + b * stride_xb + k_ids * stride_xk,
            mask=k_mask,
            other=0.0,
        )

        w_vals = tl.load(
            W_ptr + n_offs[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        prod = w_vals * x_vals[None, :]
        tmp = tl.sum(prod, axis=1)          # contribution to y1
        y1 += tmp
        s2 += tl.sum(prod * prod, axis=1)   # contribution to s2

    # t-stat like confidence
    denom = tl.sqrt(s2 / tl.maximum(K0, 1) + eps)
    tstat = tl.abs(y1) / tl.maximum(denom, eps)
    passed = (tstat < tau) & n_mask        # output lanes that pass ET
    acc = y1                               # accumulator initialized with prefix sum

    # ------------------------------------------------
    # Phase 2: tail columns [K0, K) with masking
    # ------------------------------------------------
    # still_active = lanes that have NOT passed and are in-range
    still_active = (~passed) & n_mask

    # We iterate a fixed MAX_ITERS times; each iteration handles a BLOCK_K chunk
    for it in range(0, MAX_ITERS):
        k = K0 + it * BLOCK_K
        k_ids = k + tl.arange(0, BLOCK_K)
        k_mask = k_ids < K       # only process within K

        # Is this tail chunk even relevant? If k >= K, no work.
        # But we cannot 'break', so just rely on k_mask to suppress loads.
        lane_mask = still_active

        # Load W only for active lanes & valid K
        w_vals = tl.load(
            W_ptr + n_offs[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=lane_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load X if any lane is still active and k is valid.
        any_active = tl.sum(lane_mask) > 0
        x_vals = tl.load(
            X_ptr + b * stride_xb + k_ids * stride_xk,
            mask=k_mask & any_active,
            other=0.0,
        )

        prod = w_vals * x_vals[None, :]
        partial = tl.sum(prod, axis=1)

        # Only accumulate for still-active lanes
        acc += partial * lane_mask.to(tl.float32)

    # Final output: passed lanes are zeroed (like your PyTorch code),
    # others keep y1 + tail
    out = tl.where(passed, 0.0, acc)
    tl.store(
        Y_ptr + b * stride_yb + n_offs * stride_yn,
        out,
        mask=n_mask,
    )


# ============================================================
# Python Wrapper: ET Matmul + Profiling
# ============================================================

@dataclass
class ETCallProfile:
    elapsed_seconds: float
    approx_cycles: float
    total_flops: float
    flops_linear: float
    flops_confidence: float
    est_memory_bytes: float
    passed_ratio: float


class TritonEarlyTermMatmul:
    """
    Low-level callable:

        x: (B, K)
        W: (N, K)  (stored on device)
        y = x @ W^T with early termination on K0 prefix.

    Uses `linear_etu_kernel` and also returns a profiling dict.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        early_k: int,
        tau: float,
        block_n: int = 128,
        block_k: int = 64,
        max_iters: int = 64,     # supports K up to early_k + max_iters * block_k
        eps: float = 1e-6,
    ):
        assert weight.is_cuda, "Weight must be on CUDA."
        self.weight = weight.contiguous()
        self.N, self.K = self.weight.shape
        self.K0 = early_k
        self.tau = float(tau)
        self.BLOCK_N = block_n
        self.BLOCK_K = block_k
        self.MAX_ITERS = max_iters
        self.eps = float(eps)

        max_supported_K = self.K0 + self.MAX_ITERS * self.BLOCK_K
        if self.K > max_supported_K:
            raise ValueError(
                f"K={self.K} larger than supported max K={max_supported_K}. "
                f"Increase max_iters or block_k."
            )

    @torch.no_grad()
    def __call__(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, ETCallProfile]:
        """
        x_2d: (B, K)  (last dim must match weight.K)

        Returns: (y, profile)
          y: (B, N)  in float32 (caller can cast back)
        """
        assert x_2d.is_cuda
        assert x_2d.ndim == 2 and x_2d.shape[1] == self.K

        B, K = x_2d.shape
        N = self.N

        y = torch.empty((B, N), device=x_2d.device, dtype=torch.float32)

        stride_xb, stride_xk = x_2d.stride()
        stride_wn, stride_wk = self.weight.stride()
        stride_yb, stride_yn = y.stride()

        grid = (B, triton.cdiv(N, self.BLOCK_N))


        t0 = time.perf_counter()
        nvtx.range_push('etu')
        linear_etu_kernel[grid](
            x_2d, self.weight, y,
            B, N, K,
            self.tau, self.eps,
            stride_xb, stride_xk,
            stride_wn, stride_wk,
            stride_yb, stride_yn,
            K0=self.K0,
            MAX_ITERS=self.MAX_ITERS,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        torch.cuda.synchronize()
        nvtx.range_pop()
        elapsed_s = time.perf_counter() - t0

        # Host-side estimate of passed_ratio (same confidence rule on prefix)
        with torch.no_grad():
            x0 = x_2d[:, :self.K0].float()
            w0 = self.weight[:, :self.K0].float()
            y1 = x0 @ w0.t()
            cum_sq = torch.zeros_like(y1)
            # NOTE: this is O(K0) per call; that's fine for profiling / small K0.
            for k_idx in range(self.K0):
                tmp = (
                    x_2d[:, k_idx : k_idx + 1]
                    .float()
                    @ self.weight[:, k_idx : k_idx + 1].float().t()
                )
                cum_sq += tmp.square_()
            tstat = y1.abs() / torch.sqrt(cum_sq / max(self.K0, 1) + 1e-6)
            passed_mask = tstat < self.tau
            passed_ratio = passed_mask.float().mean().item()
            remain_ratio = 1.0 - passed_ratio

        # FLOPs model: prefix (K0) + remaining scaled by remain_ratio
        macs_per_elem = self.K0 + remain_ratio * (K - self.K0)
        flops_linear = 2.0 * macs_per_elem * (B * N)            # mul+add
        flops_conf = (self.K0 * 2 + 6) * (B * N)                # rough overhead
        total_flops = flops_linear + flops_conf

        # Approx memory traffic (just for this matmul)
        bytes_per = x_2d.element_size()
        x_bytes = B * (self.K0 + remain_ratio * (K - self.K0)) * bytes_per
        w_bytes = N * (self.K0 + remain_ratio * (K - self.K0)) * bytes_per
        y_bytes = B * N * y.element_size()
        est_traffic_bytes = x_bytes + w_bytes + y_bytes

        props = torch.cuda.get_device_properties(x_2d.device)
        sm_clock_hz = getattr(props, "clockRate", None)
        if sm_clock_hz is None:
            # fallback for older or vendor-specific builds
            sm_clock_hz = getattr(props, "clock_rate", 0)
        # convert kHz → Hz
        sm_clock_hz = sm_clock_hz * 1e3

        approx_cycles = elapsed_s * sm_clock_hz

        profile = ETCallProfile(
            elapsed_seconds=elapsed_s,
            approx_cycles=approx_cycles,
            total_flops=total_flops,
            flops_linear=flops_linear,
            flops_confidence=flops_conf,
            est_memory_bytes=est_traffic_bytes,
            passed_ratio=passed_ratio,
        )
        return y, profile


# ============================================================
# nn.Linear wrapper + utils for patching a model
# ============================================================

@dataclass
class ETStatsAccum:
    calls: int = 0
    elapsed_s: float = 0.0
    est_mem_bytes: float = 0.0
    passed_weighted_sum: float = 0.0
    outputs_count: int = 0


class TritonLinearModule(nn.Module):
    """
    Drop-in replacement for nn.Linear using TritonEarlyTermMatmul.

    - Keeps original weight & bias (copied).
    - Supports arbitrary batch/sequence dims: [..., K] → [..., N].
    - Accumulates per-layer ET stats.
    """

    def __init__(
        self,
        linear: nn.Linear,
        early_k: int = 256,
        tau: float = 3.0,
        block_n: int = 128,
        block_k: int = 64,
        max_iters: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert linear.weight.is_cuda, "Move model to CUDA before patching with Triton."
        self.weight = nn.Parameter(linear.weight.detach().contiguous())
        self.bias = (
            nn.Parameter(linear.bias.detach())
            if linear.bias is not None
            else None
        )
        self.N, self.K = self.weight.shape

        self.op = TritonEarlyTermMatmul(
            self.weight,
            early_k=early_k,
            tau=tau,
            block_n=block_n,
            block_k=block_k,
            max_iters=max_iters,
            eps=eps,
        )
        self.stats = ETStatsAccum()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        assert (
            K == self.K
        ), f"Input last dim {K} != weight K {self.K} in TritonLinearModule."

        x_2d = x.reshape(-1, K).contiguous()
        y_2d, prof = self.op(x_2d)
        if self.bias is not None:
            y_2d = y_2d + self.bias

        # accumulate stats
        self.stats.calls += 1
        self.stats.elapsed_s += prof.elapsed_seconds
        self.stats.est_mem_bytes += prof.est_memory_bytes
        self.stats.passed_weighted_sum += prof.passed_ratio * y_2d.numel()
        self.stats.outputs_count += y_2d.numel()

        y = y_2d.reshape(*orig_shape[:-1], self.N)
        return y.to(x.dtype)


class MemoryTrackerModule(nn.Module):
    """
    Tracks REAL off-chip memory transfer for a Linear layer.
    Handles both 2D ([N, D]) and 3D ([B, T, D]) input.
    """
    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name
        self.bytes = 0

    def forward(self, x):
        W = self.module.weight
        D_out, D_in = W.shape
        dtype_size = x.element_size()

        # --------------------------------------------
        # CASE 1: x is [B, T, D]
        # --------------------------------------------
        if x.dim() == 3:
            B, T, D = x.shape
            assert D == D_in
            N = B * T

        # --------------------------------------------
        # CASE 2: x is [N, D] (OPT MLP case!)
        # --------------------------------------------
        elif x.dim() == 2:
            N, D = x.shape
            assert D == D_in

        # --------------------------------------------
        # Unsupported shape
        # --------------------------------------------
        else:
            raise ValueError(f"Unsupported input shape for Linear: {x.shape}")

        # -------------------------------------------------------------
        # Real off-chip memory traffic:
        #   read_X = N * D
        #   read_W = D_out * D
        #   write_Y = N * D_out
        # -------------------------------------------------------------
        read_X  = N * D_in  * dtype_size
        read_W  = D_out * D_in * dtype_size
        write_Y = N * D_out * dtype_size

        self.bytes += (read_X + read_W + write_Y)
        return self.module(x)



def patch_model_with_triton(
    model: nn.Module,
    use_triton: bool = True,
    early_k: int = 36,
    tau: float = 0.02,
    block_n: int = 128,
    block_k: int = 64,
    max_iters: int = 64,
    eps: float = 1e-6,
) -> nn.Module:
    """
    Recursively replace all nn.Linear modules with TritonLinearModule.
    You can easily add a filter to restrict to certain submodules.
    """
    tracking_handles = []
    tau = norm.ppf(tau)
    if use_triton:
        print("Using Triton ET kernel.")
        for name, module in model.named_children():

            # ONLY patch fc1 layers inside OPT decoder blocks
            if name == "fc1" and isinstance(module, nn.Linear):
                if use_triton:
                    replaced = TritonLinearModule(
                        module,
                        early_k=early_k,
                        tau=tau,
                        block_n=block_n,
                        block_k=block_k,
                        max_iters=max_iters,
                        eps=eps,
                    )
                else:
                    replaced = module

                wrapped = MemoryTrackerModule(replaced, name=f"fc1-{name}")

            else:
                    tracking_handles = patch_model_with_triton(
                    module, use_triton=use_triton,
                    early_k=early_k, tau=tau, block_n=block_n,
                    block_k=block_k, max_iters=max_iters, eps=eps
                )
    else:
        for name, module in model.named_children():
          if name == "fc1" and isinstance(module, nn.Linear):
              wrapped = MemoryTrackerModule(module, name=name)
    return tracking_handles


def collect_et_stats(model: nn.Module) -> ETStatsAccum:
    """
    Aggregate ET stats from all TritonLinearModule instances in a model.
    """
    agg = ETStatsAccum()
    for m in model.modules():
        if isinstance(m, TritonLinearModule):
            agg.calls += m.stats.calls
            agg.elapsed_s += m.stats.elapsed_s
            agg.est_mem_bytes += m.stats.est_mem_bytes
            agg.passed_weighted_sum += m.stats.passed_weighted_sum
            agg.outputs_count += m.stats.outputs_count
    return agg




# ============================================================
# Main profiling entrypoint
# ============================================================

EARLY_K = 256
TAU = 0.02        # probability → converted to z-score
BLOCK_N = 128
BLOCK_K = 64
MAX_ITERS = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--et",
        action="store_true",
        help="Use early termination Triton kernel (otherwise pure baseline matmul).",
    )
    args = parser.parse_args()

    device = "cuda"
    model_name = "facebook/opt-125m"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Patching OPT-125M Linear layers with Triton ET kernel...")
    patch_model_with_triton(
        model,
        early_k=EARLY_K,
        tau=TAU,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        max_iters=MAX_ITERS,
    )
    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    

    # Warmup (not profiled)
    print("Warmup...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)

    label = "ET" if args.et else "BASELINE"
    print("Profiling label:", label)

    # 🔴 Nsight Compute should capture this region
    nvtx.range_push("ET")
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    nvtx.range_pop()

    print("Done. Logits mean:", out.logits.float().mean().item())


if __name__ == "__main__":
    main()


