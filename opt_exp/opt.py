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
    model.eval()
    DEVICE = "cuda"
    correct = 0

    # ---- evaluate only the first 10 samples ----
    n = min(1, len(dataset))
    subset = 1 #dataset.select(range(n))

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

import torch
import triton
import triton.language as tl

_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K", "HAS_BIAS", "OUT_TL_DTYPE"],
)
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


def triton_linear_fwd(x: torch.Tensor, weight_nk: torch.Tensor, bias: torch.Tensor | None = None, out_dtype=None):
    """
    x:      [M, K]
    weight: [N, K]
    bias:   [N] or None
    returns [M, N]
    """
    assert x.is_cuda and weight_nk.is_cuda
    assert x.ndim == 2 and weight_nk.ndim == 2
    M, K = x.shape
    N, K2 = weight_nk.shape
    assert K == K2

    if out_dtype is None:
        out_dtype = x.dtype
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

    out_tl_dtype = _TORCH_TO_TL[out_dtype]

    o = torch.empty((M, N), device=x.device, dtype=out_dtype)

    # IMPORTANT: grid should match BM/BN, but autotune chooses BM/BN.
    # Use "meta" fields by giving a grid lambda so it can see the chosen BM/BN.
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))

    linear_fwd_kernel[grid](
        x, weight_nk, bias, o,
        M=M, N=N, K=K,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_wn=weight_nk.stride(0), stride_wk=weight_nk.stride(1),
        stride_om=o.stride(0), stride_on=o.stride(1),
        HAS_BIAS=(bias is not None),
        OUT_TL_DTYPE=out_tl_dtype,
    )
    return o


def opt_fc1_triton(x_bsh: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor | None):
    """
    x_bsh:      [B, S, H]
    fc1_weight: [4H, H]  (torch Linear weight)
    fc1_bias:   [4H] or None
    returns:    [B, S, 4H]
    """
    assert x_bsh.is_cuda and fc1_weight.is_cuda
    B, S, H = x_bsh.shape
    x2 = x_bsh.reshape(B * S, H).contiguous()
    y2 = triton_linear_fwd(x2, fc1_weight, fc1_bias, out_dtype=x_bsh.dtype)
    return y2.reshape(B, S, fc1_weight.shape[0])


def check_linear_match(
    x, weight, bias=None,
    force_fp32_exact_check=True,
    rtol=0.0, atol=0.0,   # for bitwise check keep 0
    tol_rtol=1e-3, tol_atol=1e-3,  # fallback tolerance for fp16/bf16
):
    """
    If force_fp32_exact_check=True, it will also test a FP32 path with TF32 disabled
    and report whether bitwise equality holds there.
    """

    assert x.is_cuda and weight.is_cuda
    torch.cuda.synchronize()

    # Important for "exact" comparisons:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Torch reference
    with torch.no_grad():
        y_ref = torch.nn.functional.linear(x, weight, bias)

        # Triton
        y_tri = triton_linear_fwd(x, weight, bias, out_dtype=y_ref.dtype)

    # Exact check (bitwise)
    exact = torch.equal(y_ref, y_tri)
    max_abs = (y_ref - y_tri).abs().max().item()
    denom = y_ref.abs().max().item()
    max_rel = (max_abs / (denom + 1e-12))

    print(f"[dtype={x.dtype}] torch.equal: {exact}")
    print(f"max_abs={max_abs:.6g}  max_rel={max_rel:.6g}")

    if not exact:
        ok_tol = torch.allclose(y_ref, y_tri, rtol=tol_rtol, atol=tol_atol)
        print(f"allclose(rtol={tol_rtol}, atol={tol_atol}): {ok_tol}")

    if force_fp32_exact_check and x.dtype != torch.float32:
        x32 = x.float()
        w32 = weight.float()
        b32 = bias.float() if bias is not None else None
        with torch.no_grad():
            y_ref32 = torch.nn.functional.linear(x32, w32, b32)
            y_tri32 = triton_linear_fwd(x32, w32, b32, out_dtype=torch.float32)
        exact32 = torch.equal(y_ref32, y_tri32)
        max_abs32 = (y_ref32 - y_tri32).abs().max().item()
        print(f"[FP32 check] torch.equal: {exact32}  max_abs={max_abs32:.6g}")

    return y_ref, y_tri


class OptFC1TritonModule(nn.Module):
    def __init__(self, torch_linear: nn.Linear, out_dtype=None):
        super().__init__()
        # keep same parameters so state_dict stays consistent
        self.weight = torch_linear.weight
        self.bias = torch_linear.bias
        self.out_dtype = out_dtype

    def forward(self, x):
        # OPT passes x as [B,S,H] into fc1
        # but to be safe, support both [B,S,H] and [M,H]
        if x.dim() == 3:
            B, S, H = x.shape
            x2 = x.reshape(B * S, H).contiguous()
            y2 = triton_linear_fwd(x2, self.weight, self.bias, out_dtype=self.out_dtype or x.dtype)
            return y2.reshape(B, S, self.weight.shape[0])
        elif x.dim() == 2:
            return triton_linear_fwd(x, self.weight, self.bias, out_dtype=self.out_dtype or x.dtype)
        else:
            raise ValueError(f"Unexpected fc1 input shape {tuple(x.shape)}")


def replace_opt_fc1_with_triton(
    model,
    verify=True,
    verify_tokens=256,
    rtol=1e-3,
    atol=1e-3,
    disable_tf32=True,
    seed=0,
):
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if dev.type != "cuda":
        raise RuntimeError("This replacement expects the model on CUDA.")

    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    replaced = 0
    checked = 0
    failed = 0

    for name, module in model.named_modules():
        if not (hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear)):
            continue

        fc1_torch: nn.Linear = module.fc1
        fc1_triton = OptFC1TritonModule(fc1_torch).to(dev)

        # ---- verify (important: compare using an input shaped like OPT uses) ----
        if verify:
            H = fc1_torch.in_features
            M = int(verify_tokens)
            # compare on 2D (matches nn.Linear) and also on 3D (matches OPT callsite)
            x2d = torch.randn((M, H), device=dev, dtype=dtype, generator=g)
            x3d = torch.randn((1, M, H), device=dev, dtype=dtype, generator=g)

            with torch.no_grad():
                y_ref2d = torch.nn.functional.linear(x2d, fc1_torch.weight, fc1_torch.bias)
                y_tri2d = fc1_triton(x2d)

                y_ref3d = torch.nn.functional.linear(x3d, fc1_torch.weight, fc1_torch.bias)
                y_tri3d = fc1_triton(x3d)

            torch.cuda.synchronize()

            def report(tag, y_ref, y_tri):
                exact = torch.equal(y_ref, y_tri)
                max_abs = (y_ref - y_tri).abs().max().item()
                denom = y_ref.abs().max().item()
                max_rel = max_abs / (denom + 1e-12)
                ok = exact or torch.allclose(y_ref, y_tri, rtol=rtol, atol=atol)
                return ok, exact, max_abs, max_rel

            ok2d, exact2d, max_abs2d, max_rel2d = report("2d", y_ref2d, y_tri2d)
            ok3d, exact3d, max_abs3d, max_rel3d = report("3d", y_ref3d, y_tri3d)

            checked += 1
            ok = ok2d and ok3d
            if not ok:
                failed += 1
                print(f"[FAIL] {name}.fc1 dtype={dtype} "
                      f"2d exact={exact2d} max_abs={max_abs2d:.3e} max_rel={max_rel2d:.3e} "
                      f"3d exact={exact3d} max_abs={max_abs3d:.3e} max_rel={max_rel3d:.3e}")
            else:
                print(f"[OK]   {name}.fc1 dtype={dtype} "
                      f"2d exact={exact2d} max_abs={max_abs2d:.3e} max_rel={max_rel2d:.3e} "
                      f"3d exact={exact3d} max_abs={max_abs3d:.3e} max_rel={max_rel3d:.3e}")

        module.fc1 = fc1_triton
        replaced += 1

    torch.cuda.synchronize()
    print(f"Replaced {replaced} OPT MLP fc1 layers with OptFC1TritonModule")
    if verify:
        print(f"Verified {checked} fc1 layers: {checked - failed} OK, {failed} FAIL (rtol={rtol}, atol={atol})")
    return model



@torch.no_grad()
def run_one_forward(model, tokenizer, device, prompt, max_new_tokens=1):
    """
    Runs a forward pass (no generation loop) and returns logits.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs)
    return out.logits, inputs


def main():
    # -------- config --------
    model_name = "facebook/opt-125m"
    device = "cuda"
    dtype = torch.float16          # use float32 if you want stricter matching
    prompt = "Hello! My name is"
    rtol, atol = 1e-3, 1e-3        # loosen/tighten depending on dtype

    # For stricter numeric comparison:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # -------- load model/tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    # -------- baseline forward --------
    logits_ref, inputs = run_one_forward(model, tokenizer, device, prompt)
    torch.cuda.synchronize()

    # -------- replace fc1 with Triton --------
    replace_opt_fc1_with_triton(
        model,
        verify=True,          # verifies each replaced fc1 against torch linear
        verify_tokens=256,
        rtol=rtol,
        atol=atol,
        disable_tf32=True,
        seed=0,
    )

    # -------- forward after replacement --------
    logits_tri, _ = run_one_forward(model, tokenizer, device, prompt)
    torch.cuda.synchronize()

    # -------- compare outputs --------
    exact = torch.equal(logits_ref, logits_tri)
    max_abs = (logits_ref - logits_tri).abs().max().item()
    denom = logits_ref.abs().max().item()
    max_rel = max_abs / (denom + 1e-12)
    allc = torch.allclose(logits_ref, logits_tri, rtol=rtol, atol=atol)

    print("\n=== FINAL LOGITS COMPARISON ===")
    print(f"torch.equal: {exact}")
    print(f"allclose(rtol={rtol}, atol={atol}): {allc}")
    print(f"max_abs={max_abs:.6g}  max_rel={max_rel:.6g}")

    # Optional: also compare argmax token at the last position (often what you care about)
    last_ref = logits_ref[:, -1, :]
    last_tri = logits_tri[:, -1, :]
    tok_ref = int(last_ref.argmax(dim=-1)[0].item())
    tok_tri = int(last_tri.argmax(dim=-1)[0].item())
    print(f"last-position argmax token id: ref={tok_ref}, triton={tok_tri}")
    if tok_ref == tok_tri:
        print("argmax token matches")
    else:
        print("argmax token differs (may still be within numeric tolerance)")

if __name__ == "__main__":
    main()
