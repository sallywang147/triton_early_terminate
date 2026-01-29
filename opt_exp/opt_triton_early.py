import torch
import triton
import triton.language as tl
import triton
import torch
import tqdm
import triton.language as tl
from datasets import load_dataset
from transformers import OPTForCausalLM, AutoTokenizer
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
from scipy.stats import norm
import torch
import torch.nn as nn
from copy import deepcopy


early_terminate_it = 32

class Confidence:
    """
    Abstract base class for negative detection methods.
    Subclasses should implement the __call__ method to determine
    if a neuron's activation is confidently negative based on statistical measures.
    """
    def __call__(self):
        """
        Determines if a neuron's activation is confidently negative based on accumulated statistics.
        
        Returns:
            Boolean tensor: True if the neuron's activation is considered confident, False otherwise
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

class StatsTestConfidence(Confidence):
    """
    Confidence class for statistical tests.
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.test_stat_bound = norm.ppf(alpha)

    def __call__(self, cum_vals: torch.Tensor, cum_squared_vals, weight, element_bias, n):
        # # Calculate mean in-place
        # cur_mean = cum_vals.div(n)
        # cur_mean.mul_(weight).add_(element_bias)
        # # Calculate variance in-place
        # cur_var = cum_squared_vals.div(n)
        # cur_var.sub_(cur_mean.pow(2))
        # cur_var.mul_(weight.pow(2))
        # # Calculate test statistic in-place
        # test_stats = cur_mean.div_(torch.sqrt_(cur_var.div_(n)))
        # res = test_stats < self.test_stat_bound
        # del cur_mean, cur_var, test_stats
        # return res
        cur_mean = cum_vals / n
        cur_mean = cur_mean * weight + element_bias
        cur_var = cum_squared_vals / n - cur_mean * cur_mean
        cur_var = cur_var * weight * weight
        test_stats = cur_mean / torch.sqrt(cur_var / n)
        res = test_stats < self.test_stat_bound
        del cur_mean, cur_var, test_stats
        return res


# linear layer with early termination, assuming followed directly by ReLU
class MyLinear(nn.Module):
    def __init__(self, linear, confidence):
        super(MyLinear, self).__init__()
        if isinstance(linear, MyLinear):
            linear = linear.linear
        self.linear = linear
        self.confidence = confidence
        self.flops = 0

    def forward(self, x):
        if self.confidence is None:
            y = self.linear(x)
            self.flops += y.numel() * self.linear.weight.shape[1] * 2 # linear layer flops
            return y
        else:
            output_channels, input_channels = self.linear.weight.shape
            y1 = x[:, :early_terminate_it] @ self.linear.weight.data[:, :early_terminate_it].T
            cum_squared = torch.zeros_like(y1)
            for c in range(early_terminate_it):
                tmp = x[:, c:c+1] @ self.linear.weight.data[:, c:c+1].T
                cum_squared += tmp.square_()
            passed_mask = self.confidence(y1, cum_squared, 1, 0, early_terminate_it)
            y2 = x[:, early_terminate_it:] @ self.linear.weight.data[:, early_terminate_it:].T
            y = torch.where(passed_mask, 0, y1+y2)
            self.flops += (passed_mask.sum().item() * early_terminate_it + (~passed_mask).sum().item() * input_channels) * 2 # linear layer flops
            self.flops += passed_mask.numel() * (early_terminate_it * 2 + 6) if isinstance(self.confidence, StatsTestConfidence) else passed_mask.numel()
            return y
    
    def gather_flops(self):
        return self.flops
        

def torch_stats_test_ref_exact(cum_vals, cum_sq, weight, element_bias, n, bound):
    cur_mean = cum_vals / n
    cur_mean = cur_mean * weight + element_bias
    cur_var = cum_sq / n - cur_mean * cur_mean
    cur_var = cur_var * weight * weight
    test_stats = cur_mean / torch.sqrt(cur_var / n)   # exact (no clamp/eps)
    return test_stats < bound                          # bool

class OptFC1EarlyTorch(nn.Module):
    """
    Pure-PyTorch early termination for OPT MLP fc1.

    Computes:
      y1      = X[:,:E] @ W[:,:E]^T (+ bias optional)
      cum_sq  = (X[:,:E]^2) @ (W[:,:E]^2)^T
      passed  = stats_test(y1, cum_sq, ...)
      y2      = X[:,E:] @ W[:,E:]^T
      y       = where(passed, 0, y1+y2)
    """
    def __init__(
        self,
        linear: nn.Linear,
        early_it: int = 32,
        test_stat_bound: float = norm.ppf(0.02),
        stats_affine_mode: str = "scalar",
        include_bias_in_y: bool = True,   # for matching OPT fc1 behavior
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.early_it = int(early_it)
        self.test_stat_bound = float(test_stat_bound)
        self.stats_affine_mode = stats_affine_mode
        self.include_bias_in_y = include_bias_in_y

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        # stats affine (buffers)
        self.register_buffer("_stats_weight_scalar", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_stats_bias_scalar", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_stats_weight_vec", torch.ones((self.out_features,), dtype=torch.float32))
        self.register_buffer("_stats_bias_vec", torch.zeros((self.out_features,), dtype=torch.float32))

    def _get_stats_affine(self, device):
        if self.stats_affine_mode == "scalar":
            w = self._stats_weight_scalar.to(device)
            b = self._stats_bias_scalar.to(device)
        elif self.stats_affine_mode == "per_col":
            w = self._stats_weight_vec.to(device)
            b = self._stats_bias_vec.to(device)
        else:
            raise ValueError("stats_affine_mode must be 'scalar' or 'per_col'")
        return w, b

    def forward(self, x):
        E = self.early_it
        W = self.weight
        b = self.bias

        orig_shape = x.shape
        if x.dim() == 3:
            B,S,H = x.shape
            x2 = x.reshape(B*S, H)
        elif x.dim() == 2:
            x2 = x
        else:
            raise ValueError(f"Unexpected x shape {orig_shape}")

        # prefix/tail
        x_e = x2[:, :E]
        w_e = W[:, :E]
        x_t = x2[:, E:]
        w_t = W[:, E:]

        # y1 and cum_sq
        y1 = x_e @ w_e.T
        cum_sq = (x_e * x_e) @ (w_e * w_e).T

        # Bias handling: OPT fc1 has bias; include it in the "real" y path
        if self.include_bias_in_y and b is not None:
            y1 = y1 + b[None, :]

        # stats
        sw, sb = self._get_stats_affine(x.device)
        passed = torch_stats_test_ref_exact(y1.float(), cum_sq.float(), sw, sb, E, self.test_stat_bound)

        # tail always computed in torch version (but masked out)
        y2 = x_t @ w_t.T
        y = torch.where(passed, torch.zeros_like(y1), y1 + y2)

        if len(orig_shape) == 3:
            return y.reshape(B, S, self.out_features)
        return y

def replace_opt_fc1_with_module(model, module_ctor):
    dev = next(model.parameters()).device
    replaced = 0
    for _, m in model.named_modules():
        if hasattr(m, "fc1") and isinstance(m.fc1, nn.Linear):
            m.fc1 = module_ctor(m.fc1).to(dev)
            replaced += 1
    torch.cuda.synchronize()
    print(f"Replaced {replaced} fc1 layers")
    return model



#NEED More revision, the acc is low 
BASE = "https://yonatanbisk.com/piqa/data"
train_dataset = load_dataset("json", data_files={"train": f"{BASE}/train.jsonl"})["train"]
tlabels = load_dataset("text", data_files={"train": f"{BASE}/train-labels.lst"})["train"]
train_dataset = train_dataset.add_column("label", [int(x["text"]) for x in tlabels])

# --- Validation split ---
valid_dataset = load_dataset("json", data_files={"validation": f"{BASE}/valid.jsonl"})["validation"]
vlabels = load_dataset("text", data_files={"validation": f"{BASE}/valid-labels.lst"})["validation"]
valid_dataset = valid_dataset.add_column("label", [int(x["text"]) for x in vlabels])

@torch.no_grad()
def eval_piqa(model, tokenizer, dataset, max_len=256):
    #model.eval()
    DEVICE = "cuda"
    correct = 0

    # ---- evaluate only the first 10 samples ----
    n = min(2000, len(dataset))
    subset = dataset.select(range(n))

    for item in tqdm.tqdm(subset, desc="Evaluating"):
        goal = item["goal"].strip()
        opts = [item["sol1"].strip(), item["sol2"].strip()]
        label = item["label"]
        losses = []
        for o in opts:
            text = f"{goal}\n{o}{tokenizer.eos_token}"
            toks = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="max_length",
            ).to(DEVICE)
            out = model(**toks, labels=toks["input_ids"])
            losses.append(out.loss.item())

        pred = int(losses[1] < losses[0])   # smaller loss = more likely
        correct += (pred == label)

    acc = correct / len(subset)
    print(f" Validation accuracy (first {len(subset)} samples): {acc:.3f}")
    return acc
_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def eval_end_to_end_three_way(
    model_name="facebook/opt-125m",
    prompts=None,
    dtype=torch.float16,
    max_len=128,
    early_it=32,
    bound=1.96,
    rtol=1e-3,
    atol=1e-3,
):
    if prompts is None:
        prompts = [
            "Hello, my name is",
            "Stanford is located in",
            "The capital of France is",
            "In a distant future, humanity",
        ]

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Baseline
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    # Torch early-terminated
    torch_et = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    replace_opt_fc1_with_module(
        torch_et,
        lambda lin: OptFC1EarlyTorch(lin, early_it=early_it, test_stat_bound=bound, stats_affine_mode="scalar", include_bias_in_y=True),
    )

    # Triton early-terminated
    triton_et = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    replace_opt_fc1_with_module(
        triton_et,
        lambda lin: OptFC1EarlyTriton(lin, early_it=early_it, test_stat_bound=bound, stats_affine_mode="scalar", include_bias_in_y=True),
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def run(model, prompt):
        inp = tok(prompt, return_tensors="pt").to(device)
        inp["input_ids"] = inp["input_ids"][:, :max_len]
        if "attention_mask" in inp:
            inp["attention_mask"] = inp["attention_mask"][:, :max_len]
        return model(**inp).logits

    stats = {
        "torch_et": {"max_abs": [], "mean_abs": [], "argmax_match": []},
        "triton_et": {"max_abs": [], "mean_abs": [], "argmax_match": []},
        "torch_vs_triton": {"max_abs": [], "mean_abs": [], "argmax_match": []},
    }

    for p in prompts:
        logits_base = run(base, p)
        logits_torch = run(torch_et, p)
        logits_triton = run(triton_et, p)
        torch.cuda.synchronize()

        # baseline comparisons
        for name, logits in [("torch_et", logits_torch), ("triton_et", logits_triton)]:
            d = (logits.float() - logits_base.float()).abs()
            stats[name]["max_abs"].append(d.max().item())
            stats[name]["mean_abs"].append(d.mean().item())

            am_base = logits_base[:, -1, :].argmax(dim=-1)
            am = logits[:, -1, :].argmax(dim=-1)
            stats[name]["argmax_match"].append((am == am_base).float().mean().item())

        # torch_et vs triton_et
        dtt = (logits_torch.float() - logits_triton.float()).abs()
        stats["torch_vs_triton"]["max_abs"].append(dtt.max().item())
        stats["torch_vs_triton"]["mean_abs"].append(dtt.mean().item())

        am_t = logits_torch[:, -1, :].argmax(dim=-1)
        am_tr = logits_triton[:, -1, :].argmax(dim=-1)
        stats["torch_vs_triton"]["argmax_match"].append((am_t == am_tr).float().mean().item())

        print(f"\nPROMPT: {p!r}")
        print(f"  torch_et  vs base:   max_abs={stats['torch_et']['max_abs'][-1]:.3e}  mean_abs={stats['torch_et']['mean_abs'][-1]:.3e}  argmax_match={stats['torch_et']['argmax_match'][-1]:.2f}")
        print(f"  triton_et vs base:   max_abs={stats['triton_et']['max_abs'][-1]:.3e}  mean_abs={stats['triton_et']['mean_abs'][-1]:.3e}  argmax_match={stats['triton_et']['argmax_match'][-1]:.2f}")
        print(f"  torch_et vs triton:  max_abs={stats['torch_vs_triton']['max_abs'][-1]:.3e}  mean_abs={stats['torch_vs_triton']['mean_abs'][-1]:.3e}  argmax_match={stats['torch_vs_triton']['argmax_match'][-1]:.2f}")

    print("\n=== SUMMARY (mean over prompts) ===")
    for k,v in stats.items():
        print(f"{k}: max_abs={sum(v['max_abs'])/len(v['max_abs']):.3e}  mean_abs={sum(v['mean_abs'])/len(v['mean_abs']):.3e}  argmax_match={sum(v['argmax_match'])/len(v['argmax_match']):.2f}")

    return stats


@triton.jit
def linear_fwd_kernel(
    X_ptr, W_ptr, B_ptr, O_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    OUT_TL_DTYPE: tl.constexpr,   # <- must be tl.float16 / tl.bfloat16 / tl.float32
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    x_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = W_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # K is constexpr, so this loop is compile-time unrolled in blocks
    for k0 in tl.static_range(0, K, BK):
        k = k0 + rk
        x = tl.load(
            x_ptrs + k0 * stride_xk,
            mask=(rm[:, None] < M) & (k[None, :] < K),
            other=0.0
        ).to(tl.float32)
        w = tl.load(
            w_ptrs + k0 * stride_wk,
            mask=(rn[None, :] < N) & (k[:, None] < K),
            other=0.0
        ).to(tl.float32)

        acc += tl.dot(x, w)

    if HAS_BIAS:
        b = tl.load(B_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += b[None, :]

    o_ptrs = O_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(o_ptrs, acc.to(OUT_TL_DTYPE), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def stats_test_kernel(
    C_ptr, CS_ptr,              # cum_vals, cum_squared_vals
    W_ptr, B_ptr,               # weight, element_bias (scalar or [N])
    OUT_ptr,                    # uint8 output (0/1), we'll view as bool
    M: tl.constexpr, N: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    STRIDE_W: tl.constexpr,     # 0 => scalar weight, 1 => per-col weight
    STRIDE_B: tl.constexpr,     # 0 => scalar bias,   1 => per-col bias
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    n: tl.constexpr,            # integer prefix length (compile-time)
    bound: tl.constexpr,        # test_stat_bound (compile-time float)
    eps: tl.constexpr,          # numerical safety (compile-time float)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    # pointers for tiles [BM, BN]
    c_ptrs  = C_ptr  + rm[:, None] * stride_cm  + rn[None, :] * stride_cn
    cs_ptrs = CS_ptr + rm[:, None] * stride_csm + rn[None, :] * stride_csn

    mask = (rm[:, None] < M) & (rn[None, :] < N)

    cum_vals = tl.load(c_ptrs,  mask=mask, other=0.0).to(tl.float32)
    cum_sq   = tl.load(cs_ptrs, mask=mask, other=0.0).to(tl.float32)

    # weight/bias: either scalar (STRIDE_*==0) or per-col (STRIDE_*==1)
    w = tl.load(W_ptr + rn * STRIDE_W, mask=(rn < N), other=0.0).to(tl.float32)  # [BN]
    b = tl.load(B_ptr + rn * STRIDE_B, mask=(rn < N), other=0.0).to(tl.float32)  # [BN]

    inv_n = 1.0 / tl.full((), n, tl.float32)

    # cur_mean = cum_vals / n
    cur_mean = cum_vals * inv_n

    # cur_mean = cur_mean * weight + element_bias
    cur_mean = cur_mean * w[None, :] + b[None, :]

    # cur_var = cum_squared_vals / n - cur_mean * cur_mean
    cur_var = cum_sq * inv_n - cur_mean * cur_mean

    # cur_var = cur_var * weight * weight
    cur_var = cur_var * (w[None, :] * w[None, :])

    # test_stats = cur_mean / sqrt(cur_var / n)
    # add eps to avoid div-by-zero or negative numerical noise
    denom = tl.sqrt(tl.maximum(cur_var * inv_n, eps))
    test_stats = cur_mean / denom

    res = test_stats < bound  # bool

    # store as uint8 (0/1)
    out_ptrs = OUT_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, res.to(tl.uint8), mask=mask)


def triton_stats_test(
    cum_vals: torch.Tensor,            # [M,N]
    cum_squared_vals: torch.Tensor,    # [M,N]
    weight: torch.Tensor,              # scalar () or [N]
    element_bias: torch.Tensor,        # scalar () or [N]
    n: int,
    test_stat_bound: float,
    eps: float = 1e-12,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
):
    assert cum_vals.is_cuda and cum_squared_vals.is_cuda
    assert cum_vals.ndim == 2 and cum_squared_vals.ndim == 2
    assert cum_vals.shape == cum_squared_vals.shape
    M, N = cum_vals.shape

    # Normalize weight/bias to CUDA tensors
    assert weight.is_cuda and element_bias.is_cuda
    if weight.ndim == 0:
        stride_w = 0
    else:
        assert weight.ndim == 1 and weight.numel() == N
        stride_w = 1

    if element_bias.ndim == 0:
        stride_b = 0
    else:
        assert element_bias.ndim == 1 and element_bias.numel() == N
        stride_b = 1

    # output stored as uint8 then viewed as bool
    out_u8 = torch.empty((M, N), device=cum_vals.device, dtype=torch.uint8)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    stats_test_kernel[grid](
        cum_vals, cum_squared_vals,
        weight, element_bias,
        out_u8,
        M=M, N=N,
        stride_cm=cum_vals.stride(0), stride_cn=cum_vals.stride(1),
        stride_csm=cum_squared_vals.stride(0), stride_csn=cum_squared_vals.stride(1),
        STRIDE_W=stride_w,
        STRIDE_B=stride_b,
        stride_om=out_u8.stride(0), stride_on=out_u8.stride(1),
        n=n,
        bound=test_stat_bound,
        eps=eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    return out_u8.bool()


@torch.no_grad()
def torch_stats_test_ref(cum_vals, cum_squared_vals, weight, element_bias, n, test_stat_bound, eps=1e-12):
    cur_mean = cum_vals / n
    cur_mean = cur_mean * weight + element_bias
    cur_var = cum_squared_vals / n - cur_mean * cur_mean
    cur_var = cur_var * weight * weight
    test_stats = cur_mean / torch.sqrt(cur_var / n + eps)
    res = test_stats < test_stat_bound
    return res

@torch.no_grad()
def check_stats_test(
    cum_vals, cum_squared_vals,
    weight, element_bias,
    n, bound,
):
    # Triton
    res_tri = triton_stats_test(cum_vals, cum_squared_vals, weight, element_bias, n, bound)
    # Torch
    res_ref = torch_stats_test_ref(cum_vals, cum_squared_vals, weight, element_bias, n, bound)

    torch.cuda.synchronize()

    exact = torch.equal(res_tri, res_ref)
    # helpful diagnostics: how many mismatches?
    mismatch = (res_tri ^ res_ref).sum().item()
    total = res_ref.numel()
    print(f"stats_test: torch.equal={exact} mismatches={mismatch}/{total}")
    return res_ref, res_tri


@triton.jit
def partial_y_and_cumsq_kernel(
    X_ptr, W_ptr,
    Y_ptr, CS_ptr,
    M: tl.constexpr, N: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    stride_csm: tl.constexpr, stride_csn: tl.constexpr,
    EARLY_IT: tl.constexpr,                # early_terminate_it (compile-time)
    OUT_TL_DTYPE: tl.constexpr,            # tl.float16 / tl.bfloat16 / tl.float32
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    x_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = W_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk

    acc_y  = tl.zeros((BM, BN), dtype=tl.float32)
    acc_cs = tl.zeros((BM, BN), dtype=tl.float32)

    # Only iterate over k in [0, EARLY_IT)
    for k0 in tl.static_range(0, EARLY_IT, BK):
        k = k0 + rk

        x = tl.load(
            x_ptrs + k0 * stride_xk,
            mask=(rm[:, None] < M) & (k[None, :] < EARLY_IT),
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            w_ptrs + k0 * stride_wk,
            mask=(rn[None, :] < N) & (k[:, None] < EARLY_IT),
            other=0.0,
        ).to(tl.float32)

        acc_y  += tl.dot(x, w)
        acc_cs += tl.dot(x * x, w * w)

    y_ptrs  = Y_ptr  + rm[:, None] * stride_ym  + rn[None, :] * stride_yn
    cs_ptrs = CS_ptr + rm[:, None] * stride_csm + rn[None, :] * stride_csn
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)

    tl.store(y_ptrs,  acc_y.to(OUT_TL_DTYPE),  mask=mask_out)
    tl.store(cs_ptrs, acc_cs.to(OUT_TL_DTYPE), mask=mask_out)


def triton_partial_y_and_cumsq(
    x_mk: torch.Tensor,         # [M, K_full]
    w_nk: torch.Tensor,         # [N, K_full] (torch Linear weight layout)
    early_terminate_it: int = 32,
    out_dtype=None,             # defaults to x.dtype
    BM=128, BN=128, BK=32,
    num_warps=8, num_stages=3,
):
    """
    Returns:
      y1:         [M, N]  (X[:,:E] @ W[:,:E]^T)
      cum_squared:[M, N]  (sum_c (X[:,c] W[:,c]^T)^2) == (X^2) @ (W^2)^T over first E cols
    """
    assert x_mk.is_cuda and w_nk.is_cuda
    assert x_mk.ndim == 2 and w_nk.ndim == 2
    M, K = x_mk.shape
    N, K2 = w_nk.shape
    assert K == K2
    assert 0 < early_terminate_it <= K

    if out_dtype is None:
        out_dtype = x_mk.dtype
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")
    out_tl_dtype = _TORCH_TO_TL[out_dtype]

    # If using tl.static_range(0, EARLY_IT, BK), EARLY_IT should be multiple of BK
    if early_terminate_it % BK != 0:
        raise ValueError(f"early_terminate_it ({early_terminate_it}) must be a multiple of BK ({BK}) for this kernel.")

    y  = torch.empty((M, N), device=x_mk.device, dtype=out_dtype)
    cs = torch.empty((M, N), device=x_mk.device, dtype=out_dtype)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    partial_y_and_cumsq_kernel[grid](
        x_mk, w_nk, y, cs,
        M=M, N=N,
        stride_xm=x_mk.stride(0), stride_xk=x_mk.stride(1),
        stride_wn=w_nk.stride(0), stride_wk=w_nk.stride(1),
        stride_ym=y.stride(0),    stride_yn=y.stride(1),
        stride_csm=cs.stride(0),  stride_csn=cs.stride(1),
        EARLY_IT=early_terminate_it,
        OUT_TL_DTYPE=out_tl_dtype,
        BM=BM, BN=BN, BK=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y, cs


@torch.no_grad()
def torch_partial_y_and_cumsq_ref(x_mk, w_nk, early_terminate_it=32):
    x_e = x_mk[:, :early_terminate_it]
    w_e = w_nk[:, :early_terminate_it]
    y1 = x_e @ w_e.T
    cum_squared = (x_e * x_e) @ (w_e * w_e).T
    return y1, cum_squared


@torch.no_grad()
def check_partial_outputs(x_mk, w_nk, early_terminate_it=32, out_dtype=None, rtol=1e-3, atol=1e-3):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    y_ref, cs_ref = torch_partial_y_and_cumsq_ref(x_mk, w_nk, early_terminate_it)
    y_tri, cs_tri = triton_partial_y_and_cumsq(x_mk, w_nk, early_terminate_it, out_dtype=out_dtype)

    torch.cuda.synchronize()

    # compare in fp32 for stability if desired
    y_ok = torch.allclose(y_tri.float(), y_ref.float(), rtol=rtol, atol=atol)
    cs_ok = torch.allclose(cs_tri.float(), cs_ref.float(), rtol=rtol, atol=atol)

    print(f"y1  allclose={y_ok}  max_abs={(y_tri.float()-y_ref.float()).abs().max().item():.3e}")
    print(f"cs  allclose={cs_ok} max_abs={(cs_tri.float()-cs_ref.float()).abs().max().item():.3e}")
    return (y_ref, cs_ref), (y_tri, cs_tri)


# =========================
# Example usage (END-TO-END)
#   1) run partial_y_and_cumsq on an OPT model's real fc1 inputs (captured via hooks)
#   2) run stats_test on the resulting (y1, cum_squared)
#   3) compare both Triton outputs vs Torch reference
# =========================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assumes you already defined:
#   triton_partial_y_and_cumsq, torch_partial_y_and_cumsq_ref
# and you already defined (from earlier):
#   triton_stats_test, torch_stats_test_ref

@torch.no_grad()
def check_opt_partial_then_stats_triton_vs_torch(
    model,
    tokenizer,
    prompt: str,
    early_terminate_it: int = 32,
    max_positions: int = 128,
    layers_to_check=None,

    # partial check tolerances
    rtol_partial: float = 1e-3,
    atol_partial: float = 1e-3,

    # stats_test params
    n: int | None = None,               # if None, uses early_terminate_it
    test_stat_bound: float = norm.ppf(0.02),
    eps: float = 1e-12,

    # weight/bias mode for stats_test
    # "scalar" => single scalar weight/bias
    # "per_col" => [N] weight/bias for the layer
    stats_affine_mode: str = "scalar",

    out_dtype=None,                     # for partial kernel outputs; defaults to x dtype
    include_bias_in_y1: bool = False,   # match your earlier spec by default (no bias)
):
    model.eval()
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError("Model must be on CUDA for Triton checks.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    layers = model.model.decoder.layers
    L = len(layers)
    if layers_to_check is None:
        layers_to_check = list(range(L))
    else:
        layers_to_check = list(layers_to_check)

    captured = {}
    hooks = []

    for i in layers_to_check:
        fc1 = layers[i].fc1

        def make_hook(layer_idx):
            def hook(mod, inputs, output):
                captured[layer_idx] = inputs[0].detach()
            return hook

        hooks.append(fc1.register_forward_hook(make_hook(i)))

    # forward to populate captured inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    if inputs["input_ids"].shape[1] > max_positions:
        inputs["input_ids"] = inputs["input_ids"][:, :max_positions]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_positions]

    _ = model(**inputs)
    torch.cuda.synchronize()

    for h in hooks:
        h.remove()

    n_eff = int(n if n is not None else early_terminate_it)

    total_fail_partial = 0
    total_fail_stats = 0

    for i in layers_to_check:
        if i not in captured:
            print(f"[WARN] layer={i:02d} no captured fc1 input")
            continue

        x = captured[i]                  # could be 2D [M,H] or 3D [B,S,H]
        fc1 = layers[i].fc1
        W = fc1.weight.detach()          # [N,H]
        b_fc1 = fc1.bias.detach() if fc1.bias is not None else None

        # normalize x to [M,H]
        if x.dim() == 3:
            B, S, H = x.shape
            x2 = x.reshape(B * S, H).contiguous()
            shape_tag = f"x=[{B},{S},{H}]"
        elif x.dim() == 2:
            M, H = x.shape
            x2 = x.contiguous()
            shape_tag = f"x=[{M},{H}]"
        else:
            raise ValueError(f"Unexpected fc1 input dim={x.dim()} shape={tuple(x.shape)} at layer={i}")

        N, H2 = W.shape
        assert H == H2
        E = early_terminate_it
        assert 0 < E <= H

        # -------------------------------
        # (A) Partial kernel: y1 + cum_sq
        # -------------------------------
        # Torch reference
        y_ref, cs_ref = torch_partial_y_and_cumsq_ref(x2, W, early_terminate_it=E)
        if include_bias_in_y1 and b_fc1 is not None:
            y_ref = y_ref + b_fc1[None, :]

        # Triton
        y_tri, cs_tri = triton_partial_y_and_cumsq(
            x2, W,
            early_terminate_it=E,
            out_dtype=(out_dtype or x2.dtype),
            BM=128, BN=128, BK=32,
            num_warps=8, num_stages=3,
        )
        if include_bias_in_y1 and b_fc1 is not None:
            y_tri = y_tri + b_fc1[None, :].to(y_tri.dtype)

        torch.cuda.synchronize()

        y_ok = torch.allclose(y_tri.float(), y_ref.float(), rtol=rtol_partial, atol=atol_partial)
        cs_ok = torch.allclose(cs_tri.float(), cs_ref.float(), rtol=rtol_partial, atol=atol_partial)
        y_max = (y_tri.float() - y_ref.float()).abs().max().item()
        cs_max = (cs_tri.float() - cs_ref.float()).abs().max().item()

        partial_status = "OK" if (y_ok and cs_ok) else "FAIL"
        if partial_status == "FAIL":
            total_fail_partial += 1

        # --------------------------------
        # (B) Stats test: res = (test < bound)
        # --------------------------------
        # choose weight/bias for stats_test
        if stats_affine_mode == "scalar":
            weight = torch.tensor(1.0, device=dev, dtype=torch.float32)
            element_bias = torch.tensor(0.0, device=dev, dtype=torch.float32)
        elif stats_affine_mode == "per_col":
            weight = torch.ones((N,), device=dev, dtype=torch.float32)
            element_bias = torch.zeros((N,), device=dev, dtype=torch.float32)
        else:
            raise ValueError("stats_affine_mode must be 'scalar' or 'per_col'")

        # Torch stats reference (use float32 for stability)
        res_ref = torch_stats_test_ref(
            y_ref.float(), cs_ref.float(),
            weight, element_bias,
            n_eff, test_stat_bound, eps=eps
        )

        # Triton stats (also float32 inputs)
        res_tri = triton_stats_test(
            y_tri.float(), cs_tri.float(),
            weight, element_bias,
            n_eff, test_stat_bound, eps=eps
        )

        torch.cuda.synchronize()

        stats_exact = torch.equal(res_tri, res_ref)
        mism = (res_tri ^ res_ref).sum().item()
        stats_status = "OK" if stats_exact else "FAIL"
        if stats_status == "FAIL":
            total_fail_stats += 1

        print(
            f"[layer={i:02d}] {shape_tag} W=[{N},{H}] E={E} | "
            f"partial={partial_status} y_max={y_max:.3e} cs_max={cs_max:.3e} | "
            f"stats={stats_status} mism={mism}"
        )

    print(f"\nSummary:")
    print(f"  Partial failed layers: {total_fail_partial}/{len(layers_to_check)}")
    print(f"  Stats  failed layers: {total_fail_stats}/{len(layers_to_check)}")


@torch.no_grad()
def torch_gated_output_ref(
    x_mk: torch.Tensor,      # [M, K]
    w_nk: torch.Tensor,      # [N, K]
    y1_mn: torch.Tensor,     # [M, N]
    passed_mask_mn: torch.Tensor,  # [M, N] bool (True => output 0)
    E: int,
):
    """
    Exact reference:
        y2 = x[:, E:] @ W[:, E:].T
        y  = torch.where(passed_mask, 0, y1 + y2)
    """
    y2 = x_mk[:, E:] @ w_nk[:, E:].T
    y_ref = torch.where(passed_mask_mn, torch.zeros_like(y1_mn), y1_mn + y2)
    return y_ref


@torch.no_grad()
def check_y_triton_vs_torch(
    x_mk,
    w_nk,
    y1_mn,
    passed_mask_mn,
    y_triton,
    E,
    rtol=1e-3,
    atol=1e-3,
):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Torch reference
    y_ref = torch_gated_output_ref(
        x_mk, w_nk, y1_mn, passed_mask_mn, E
    )

    torch.cuda.synchronize()

    # Compare
    exact = torch.equal(y_triton, y_ref)
    allclose = torch.allclose(y_triton, y_ref, rtol=rtol, atol=atol)

    diff = (y_triton - y_ref).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (y_ref.abs() + 1e-12)).max().item()

    mism = (diff > atol + rtol * y_ref.abs()).sum().item()

    print("=== Y CHECK (Triton vs Torch) ===")
    print(f"exact:     {exact}")
    print(f"allclose:  {allclose}")
    print(f"max_abs:   {max_abs:.3e}")
    print(f"max_rel:   {max_rel:.3e}")
    print(f"mismatch count: {mism}/{y_ref.numel()}")

    return y_ref, y_triton

# assumes you already have:
# - triton_partial_y_and_cumsq(x_mk, w_nk, early_terminate_it, ...)
# - triton_stats_test_exact(cum_vals, cum_sq, weight, bias, n, bound)
# - triton_gated_tail_add(x_mk, w_nk, y1_mn, passed_mask_mn, early_terminate_it, ...)
#
# and you want Torch stats exactly as your python:
def torch_stats_test_ref_exact(cum_vals, cum_sq, weight, element_bias, n, bound):
    cur_mean = cum_vals / n
    cur_mean = cur_mean * weight + element_bias
    cur_var = cum_sq / n - cur_mean * cur_mean
    cur_var = cur_var * weight * weight
    test_stats = cur_mean / torch.sqrt(cur_var / n)     # NO eps, NO clamp
    return test_stats < bound

@torch.no_grad()
def check_opt_layerwise_gated_tail_triton_vs_torch(
    model,
    tokenizer,
    prompt: str,
    E: int = 32,
    test_stat_bound: float = norm.ppf(0.02),
    stats_affine_mode: str = "scalar",  # "scalar" or "per_col"
    max_positions: int = 128,
    layers_to_check=None,
    # tolerances for final y comparison (fp16/bf16)
    rtol_y: float = 1e-3,
    atol_y: float = 1e-3,
    out_dtype=None,  # for Triton outputs; default uses x dtype
):
    model.eval()
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError("Model must be on CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    layers = model.model.decoder.layers
    L = len(layers)
    if layers_to_check is None:
        layers_to_check = list(range(L))
    else:
        layers_to_check = list(layers_to_check)

    # ---- capture fc1 inputs per layer ----
    captured = {}
    hooks = []
    for i in layers_to_check:
        fc1 = layers[i].fc1

        def make_hook(layer_idx):
            def hook(mod, inputs, output):
                captured[layer_idx] = inputs[0].detach()
            return hook

        hooks.append(fc1.register_forward_hook(make_hook(i)))

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    if inputs["input_ids"].shape[1] > max_positions:
        inputs["input_ids"] = inputs["input_ids"][:, :max_positions]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_positions]

    _ = model(**inputs)
    torch.cuda.synchronize()

    for h in hooks:
        h.remove()

    fail_layers = 0

    for i in layers_to_check:
        if i not in captured:
            print(f"[WARN] layer={i:02d} no captured fc1 input")
            continue

        x = captured[i]
        fc1 = layers[i].fc1
        W = fc1.weight.detach()     # [N,H]
        N, H = W.shape
        assert E <= H

        # normalize x to [M,H]
        if x.dim() == 3:
            B, S, Hx = x.shape
            assert Hx == H
            x2 = x.reshape(B * S, H).contiguous()
            shape_tag = f"x=[{B},{S},{H}]"
        elif x.dim() == 2:
            M, Hx = x.shape
            assert Hx == H
            x2 = x.contiguous()
            shape_tag = f"x=[{M},{H}]"
        else:
            raise ValueError(f"Unexpected fc1 input shape at layer {i}: {tuple(x.shape)}")

        M = x2.shape[0]

        # ---- choose stats affine params ----
        if stats_affine_mode == "scalar":
            weight = torch.tensor(1.0, device=dev, dtype=torch.float32)
            bias   = torch.tensor(0.0, device=dev, dtype=torch.float32)
        elif stats_affine_mode == "per_col":
            weight = torch.ones((N,), device=dev, dtype=torch.float32)
            bias   = torch.zeros((N,), device=dev, dtype=torch.float32)
        else:
            raise ValueError("stats_affine_mode must be 'scalar' or 'per_col'")

        # =========================
        # Torch reference pipeline
        # =========================
        x_e = x2[:, :E]
        w_e = W[:, :E]
        x_t = x2[:, E:]
        w_t = W[:, E:]

        y1_ref = x_e @ w_e.T
        cs_ref = (x_e * x_e) @ (w_e * w_e).T
        passed_ref = torch_stats_test_ref_exact(y1_ref.float(), cs_ref.float(), weight, bias, E, test_stat_bound)
        y2_ref = x_t @ w_t.T
        y_ref = torch.where(passed_ref, torch.zeros_like(y1_ref), y1_ref + y2_ref)

        # =========================
        # Triton pipeline
        # =========================
        y1_tri, cs_tri = triton_partial_y_and_cumsq(
            x2, W,
            early_terminate_it=E,
            out_dtype=(out_dtype or x2.dtype),
            BM=128, BN=128, BK=32,
            num_warps=8, num_stages=3,
        )

        _ = triton_stats_test(
            y1_tri.float(), cs_tri.float(),
            weight, bias,
            n=E,
            test_stat_bound=test_stat_bound,
        )


def triton_linear_fwd_dense(x_mk, w_nk, bias=None, out_dtype=None, BM=128, BN=128, BK=32):
    assert x_mk.is_cuda and w_nk.is_cuda
    M, K = x_mk.shape
    N, K2 = w_nk.shape
    assert K == K2

    if out_dtype is None:
        out_dtype = x_mk.dtype
    out_tl = _TORCH_TO_TL[out_dtype]

    o = torch.empty((M, N), device=x_mk.device, dtype=out_dtype)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    linear_fwd_kernel[grid](
        x_mk, w_nk, bias, o,
        M=M, N=N, K=K,
        stride_xm=x_mk.stride(0), stride_xk=x_mk.stride(1),
        stride_wn=w_nk.stride(0), stride_wk=w_nk.stride(1),
        stride_om=o.stride(0),    stride_on=o.stride(1),
        HAS_BIAS=(bias is not None),
        OUT_TL_DTYPE=out_tl,
        BM=BM, BN=BN, BK=BK,
        num_warps=8, num_stages=3,
    )
    return o


class OptFC1EarlyTriton(nn.Module):
    """
    Drop-in replacement for nn.Linear used as OPT MLP fc1.
    Computes:
      y1, cum_sq  from prefix [:E]
      passed_mask from stats test
      y = where(passed, 0, y1 + (tail GEMM))
    Returns y with the same dtype as input (by default).
    """

    def __init__(
        self,
        linear: nn.Linear,
        early_it: int = 32,
        test_stat_bound: float = norm.ppf(0.02),
        stats_affine_mode: str = "scalar",  # "scalar" or "per_col"
        out_dtype=None,                     # None => use input dtype
        # GEMM tiling (match your kernels)
        BM=128, BN=128, BK=32,
        num_warps=8, num_stages=3,
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.early_it = int(early_it)
        self.test_stat_bound = float(test_stat_bound)
        self.stats_affine_mode = stats_affine_mode
        self.out_dtype = out_dtype

        self.BM, self.BN, self.BK = BM, BN, BK
        self.num_warps, self.num_stages = num_warps, num_stages

        # store parameters (clone weights/bias into Parameters so model.state_dict works)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        # stats affine parameters (you can later learn these if you want)
        self.register_buffer("_stats_weight_scalar", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_stats_bias_scalar", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_stats_weight_vec", torch.ones((self.out_features,), dtype=torch.float32))
        self.register_buffer("_stats_bias_vec", torch.zeros((self.out_features,), dtype=torch.float32))

    def _get_stats_affine(self, device):
        if self.stats_affine_mode == "scalar":
            w = self._stats_weight_scalar.to(device)
            b = self._stats_bias_scalar.to(device)
        elif self.stats_affine_mode == "per_col":
            w = self._stats_weight_vec.to(device)
            b = self._stats_bias_vec.to(device)
        else:
            raise ValueError("stats_affine_mode must be 'scalar' or 'per_col'")
        return w, b

    def forward(self, x):
        """
        x can be [B,S,H] or [M,H].
        Returns same shape with last dim = out_features.
        """
        assert x.is_cuda, "This module expects CUDA tensors"
        E = self.early_it
        W = self.weight
        b = self.bias
        out_dtype = self.out_dtype or x.dtype

        # normalize to 2D
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, H = x.shape
            x2 = x.reshape(B * S, H).contiguous()
        elif x.dim() == 2:
            x2 = x.contiguous()
            H = x2.shape[1]
        else:
            raise ValueError(f"Unexpected x dim: {x.dim()} shape={orig_shape}")

        assert H == W.shape[1], f"Shape mismatch: x H={H}, W H={W.shape[1]}"
        assert 0 < E <= H

        # 1) prefix partial outputs
        y1, cum_sq = triton_partial_y_and_cumsq(
            x2, W,
            early_terminate_it=E,
            out_dtype=out_dtype,
            BM=self.BM, BN=self.BN, BK=self.BK,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # Optional: include bias into y1 if you want fc1 output to include bias
        # NOTE: your early-termination math earlier ignored bias; for OPT correctness you
        # usually want the REAL fc1 output (with bias).
        if b is not None:
            y1 = y1 + b[None, :].to(y1.dtype)

        # 2) stats test
        w_aff, b_aff = self._get_stats_affine(x2.device)
        passed_mask = triton_stats_test(
            y1.float(), cum_sq.float(),
            w_aff, b_aff,
            n=E,
            test_stat_bound=self.test_stat_bound,
        )  # bool [M,N]
        need_tail_row = (~passed_mask).any(dim=1)      # [M] bool
        idx = need_tail_row.nonzero(as_tuple=False).squeeze(1)  # [M2]
        M2 = idx.numel()
        # x: [M,K]
        x_tail = x[:, E:]               # [M, K_tail]
        x_tail_compact = x_tail.index_select(0, idx).contiguous()  # [M2, K_tail]
        # W: [N,K]
        W_tail = W[:, E:].contiguous()  # [N, K_tail]
    
        y2_compact = triton_linear_fwd_dense(
            x_tail_compact, W_tail, bias=None, out_dtype=y1.dtype,
            BM=128, BN=128, BK=32
        )  # [M2, N]
        torch.cuda.synchronize()
        y = torch.zeros((x.shape[0], W.shape[0]), device=x.device, dtype=y1.dtype)  # [M,N]
        y.index_copy_(0, idx, y2_compact)


        # reshape back
        if len(orig_shape) == 3:
            return y.reshape(B, S, self.out_features)
        else:
            return y

def replace_opt_fc1_with_early_triton(
    model,
    early_it=32,
    test_stat_bound=norm.ppf(0),
    stats_affine_mode="scalar",
    BM=128, BN=128, BK=32,
    num_warps=8, num_stages=3,
):
    dev = next(model.parameters()).device
    replaced = 0

    for i, layer in enumerate(model.model.decoder.layers):
            layer.fc1 = OptFC1EarlyTriton(
                layer.fc1,
                early_it=early_it,
                test_stat_bound=test_stat_bound,
                stats_affine_mode=stats_affine_mode,
                out_dtype=None,  # use input dtype
                BM=BM, BN=BN, BK=BK,
                num_warps=num_warps, num_stages=num_stages,
            ).to(dev)
            torch.cuda.synchronize()
            replaced += 1

    print(f"Replaced {replaced} fc1 layers with OptFC1EarlyTriton")
    return model

def replace_opt_fc1_with_torch_mylnear(
    model,
    early_it=32,
    test_stat_bound=1,
    stats_affine_mode="scalar",
    BM=128, BN=128, BK=32,
    num_warps=8, num_stages=3,
):
    dev = next(model.parameters()).device
    replaced = 0

    for i, layer in enumerate(model.model.decoder.layers):
        layer.fc1 = MyLinear(layer.fc1, StatsTestConfidence(0.02))
        replaced += 1

    torch.cuda.synchronize()
    print(f"Replaced {replaced} fc1 layers with OptFC1EarlyTriton")
    return model



@torch.no_grad()
def check_replacement(
    model,
    early_it=32,
    test_stat_bound=norm.ppf(0.02),
    stats_affine_mode="scalar",
    BM=128, BN=128, BK=32,
    num_warps=8, num_stages=3,
    n_tests=5,
    atol=1e-3,
    rtol=1e-3,
    verbose=True,
):
    """
    Compares OptFC1EarlyTriton(fc1) vs MyLinear(fc1, StatsTestConfidence) on random inputs
    at each OPT layer fc1 it finds.

    Returns a dict with per-layer max/mean abs diff, and overall pass/fail.
    """
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    results = {}
    replaced = 0
    all_ok = True

    for name, module in model.named_modules():
        # OPT decoder layer modules have fc1 = nn.Linear
        if hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear):
            fc1 = module.fc1
            replaced += 1

            # Build implementations
            triton_fc1 = OptFC1EarlyTriton(
                fc1,
                early_it=early_it,
                test_stat_bound=test_stat_bound,
                stats_affine_mode=stats_affine_mode,
                out_dtype=None,  # use input dtype
                BM=BM, BN=BN, BK=BK,
                num_warps=num_warps, num_stages=num_stages,
            ).to(dev)

            torch_fc1 = MyLinear(
                fc1,
                confidence=StatsTestConfidence(0.02),
            ).to(dev)

            # Ensure they don't update params / grads
            triton_fc1.eval()
            torch_fc1.eval()

            # Choose an input shape consistent with OPT MLP:
            # fc1: hidden_size -> ffn_dim, and it receives (B, T, hidden_size)
            H = fc1.in_features

            # run multiple random trials
            layer_max = 0.0
            layer_mean = 0.0
            layer_ok = True

            for t in range(n_tests):
                B = 2
                T = 16
                x = torch.randn((B*T, H), device=dev, dtype=dtype)

                # both should accept (B,T,H); if your implementations only accept 2D,
                # flatten here and reshape back.
                y_triton = triton_fc1(x)
                y_torch  = torch_fc1(x)

                # compare
                diff = (y_triton - y_torch).abs()
                max_abs = diff.max().item()
                mean_abs = diff.mean().item()

                layer_max = max(layer_max, max_abs)
                layer_mean += mean_abs

                ok = torch.allclose(y_triton, y_torch, atol=atol, rtol=rtol)
                layer_ok = layer_ok and ok
                all_ok = all_ok and ok

                if verbose and (not ok):
                    # show a bit more detail for debugging
                    idx = diff.view(-1).argmax().item()
                    vt = y_triton.view(-1)[idx].item()
                    vo = y_torch.view(-1)[idx].item()
                    print(f"[FAIL] {name}.fc1 trial {t}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                          f"worst(triton={vt:.6g}, torch={vo:.6g})")

            layer_mean /= float(n_tests)

            results[f"{name}.fc1"] = {
                "ok": layer_ok,
                "max_abs": layer_max,
                "mean_abs": layer_mean,
                "in_features": H,
                "out_features": fc1.out_features,
                "early_it": early_it,
                "BM": BM, "BN": BN, "BK": BK,
                "num_warps": num_warps, "num_stages": num_stages,
            }

            if verbose and layer_ok:
                print(f"[OK]   {name}.fc1: max_abs={layer_max:.6g} mean_abs={layer_mean:.6g}")

    if replaced == 0:
        raise RuntimeError("No modules with an nn.Linear fc1 were found. Are you sure this is an OPT model?")

    if verbose:
        print(f"\nChecked {replaced} fc1 layers. Overall: {'PASS' if all_ok else 'FAIL'}")

    return {"overall_ok": all_ok, "n_layers": replaced, "results": results}

def main():
    model_name = "facebook/opt-125m"
    device = "cuda"
    dtype = torch.float16
    prompt = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    #check_replacement(model)

    model1 = replace_opt_fc1_with_early_triton(model)
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model2 = replace_opt_fc1_with_torch_mylnear(model)
    acc1 = eval_piqa(model1, tokenizer, valid_dataset)
    print("PIQA validation acc of triton early termination:", acc1)
    acc2 = eval_piqa(model2, tokenizer, valid_dataset)
    print("PIQA validation acc torch early termination:", acc2)


if __name__ == "__main__":
    main()