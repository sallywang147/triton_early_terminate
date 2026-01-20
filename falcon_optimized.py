import triton
import torch
import tqdm
import triton.language as tl
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

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
    n = min(10, len(dataset))
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


@triton.jit
def linear_etu_kernel_row(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    K0: tl.constexpr,
    TAU: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # row in batch
    pid_n = tl.program_id(1)   # output block

    # row index
    m = pid_m
    m_mask = m < B

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < DOUT

    # accumulators per output lane (1 row)
    y1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    s2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # ----- prefix K0 -----
    for k0 in range(0, K0, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = (k_ids < DIN) & (k_ids < K0)

        x = tl.load(
            X_ptr + m * stride_xm + k_ids * stride_xk,
            mask=m_mask & k_mask,
            other=0.0,
        ).to(tl.float16)  # [BK]

        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)  # [BN, BK]

        prod = w * x[None, :]                  # [BN, BK]
        contrib = tl.sum(prod, axis=1)         # [BN]
        y1 += contrib
        s2 += tl.sum(prod * prod, axis=1)

    denom = tl.sqrt(s2 / (K0 + EPS) + EPS)
    tstat = tl.abs(y1) / tl.maximum(denom, EPS)

    passed = (tstat < TAU) & n_mask
    acc = y1

    # ----- tail DIN -----
    still_active = (~passed) & n_mask
    for k0 in range(K0, DIN, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < DIN

        # load x once per row for this k tile
        x = tl.load(
            X_ptr + m * stride_xm + k_ids * stride_xk,
            mask=m_mask & k_mask,
            other=0.0,
        ).to(tl.float16)

        # load w for this output block and k tile
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk,
            mask=still_active[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        contrib = tl.sum(w * x[None, :], axis=1).to(tl.float32)
        acc += contrib * still_active.to(tl.float32)

    # bias + zero-out passed
    bias = tl.load(B_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias
    out = acc #tl.where(passed, 0.0, acc)

    tl.store(
        Y_ptr + m * stride_ym + offs_n * stride_yn,
        out,
        mask=m_mask & n_mask,
    )


class TritonLinearETRow(nn.Module):
    def __init__(self, linear: nn.Linear, BN=128, BK=32, K0=16, tau=-2.053748910631823, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().contiguous(), requires_grad=False)

        # LLaMA often has bias=None. Use a persistent zero bias buffer.
        if linear.bias is None:
            self.register_buffer("bias_buf", torch.zeros((linear.out_features,), dtype=self.weight.dtype))
        else:
            self.register_buffer("bias_buf", linear.bias.detach().contiguous())

        self.DIN = linear.in_features
        self.DOUT = linear.out_features
        self.BN = BN
        self.BK = BK
        self.K0 = K0
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        orig_dtype = x.dtype
        orig_shape = x.shape

        if x.ndim == 3:
            x2 = x.reshape(-1, orig_shape[-1])
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        Bsz, DIN = x2.shape
        assert DIN == self.DIN, (DIN, self.DIN)

        y2 = torch.empty((Bsz, self.DOUT), device=x.device, dtype=torch.float32)

        bias = self.bias_buf
        if bias.device != x.device:
            bias = bias.to(x.device)

        grid = (Bsz, triton.cdiv(self.DOUT, self.BN))
        linear_etu_kernel_row[grid](
            x2, self.weight, bias, y2,
            Bsz, DIN, self.DOUT,
            x2.stride(0), x2.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y2.stride(0), y2.stride(1),
            K0=self.K0,
            TAU=self.tau,
            EPS=self.eps,
            BLOCK_N=self.BN,
            BLOCK_K=self.BK,
            num_warps=4,
            num_stages=3,
        )

        # return same dtype as model expects
        y2 = y2.to(orig_dtype)

        if x.ndim == 3:
            return y2.reshape(orig_shape[0], orig_shape[1], self.DOUT)
        return y2



def replace_falcon_qkv_and_up_with_triton_et(
    model,
    BN=128, BK=32, K0=16, tau=-2.053748910631823, eps=1e-6,
    num_warps=4, num_stages=3
):
    dev = next(model.parameters()).device
    counts = {"qkv": 0, "q": 0, "k": 0, "v": 0, "up": 0}

    for _, module in model.named_modules():
        # Fused QKV (most common Falcon)
        if hasattr(module, "query_key_value") and isinstance(getattr(module, "query_key_value"), nn.Linear):
            module.query_key_value = TritonLinearETRow(
                module.query_key_value,
                BN=BN, BK=BK, K0=K0, tau=tau, eps=eps,
                num_warps=num_warps, num_stages=num_stages
            ).to(dev)
            counts["qkv"] += 1

        # Optional: if a variant has explicit q/k/v
        if hasattr(module, "q_proj") and isinstance(getattr(module, "q_proj"), nn.Linear):
            module.q_proj = TritonLinearETRow(module.q_proj, BN=BN, BK=BK, K0=K0, tau=tau, eps=eps,
                                              num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["q"] += 1
        if hasattr(module, "k_proj") and isinstance(getattr(module, "k_proj"), nn.Linear):
            module.k_proj = TritonLinearETRow(module.k_proj, BN=BN, BK=BK, K0=K0, tau=tau, eps=eps,
                                              num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["k"] += 1
        if hasattr(module, "v_proj") and isinstance(getattr(module, "v_proj"), nn.Linear):
            module.v_proj = TritonLinearETRow(module.v_proj, BN=BN, BK=BK, K0=K0, tau=tau, eps=eps,
                                              num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["v"] += 1

        # MLP up projection
        if hasattr(module, "dense_h_to_4h") and isinstance(getattr(module, "dense_h_to_4h"), nn.Linear):
            module.dense_h_to_4h = TritonLinearETRow(
                module.dense_h_to_4h,
                BN=BN, BK=BK, K0=K0, tau=tau, eps=eps,
                num_warps=num_warps, num_stages=num_stages
            ).to(dev)
            counts["up"] += 1

    print("Replaced Falcon projections with TritonLinearETRow:", counts)
    return model

def main():
    assert torch.cuda.is_available()
    device = "cuda"
    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    replace_falcon_qkv_and_up_with_triton_et(model, BN=128, BK=32, K0=16, tau=-2.053748910631823)
    torch.cuda.synchronize()
    # Evaluate first 10 PIQA samples (your existing eval)
    acc = eval_piqa(model, tokenizer, valid_dataset)
    print("PIQA validation acc (first 10):", acc)

if __name__ == "__main__":
    main()