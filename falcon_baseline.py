import triton
import torch
import tqdm
import triton.language as tl
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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
def linear_baseline_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    B, DIN, DOUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    MAX_K_TILES: tl.constexpr,
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

class TritonLinearBaseline(nn.Module):
    """
    Baseline Triton linear for inference.
    Supports Linear layers with bias or without bias (LLaMA uses bias=False).
    Works with 2D [N, DIN] and 3D [B, S, DIN] by flattening.
    Returns output in the same dtype as input (fp16) to keep downstream layers happy.
    """
    def __init__(self, linear: nn.Linear, BM=32, BN=64, BK=32, num_warps=4, num_stages=3):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().contiguous(), requires_grad=False)  # [DOUT, DIN]

        # LLaMA typically has bias=None. Use a persistent zero bias buffer.
        if linear.bias is None:
            self.register_buffer("bias_buf", torch.zeros((linear.out_features,), dtype=self.weight.dtype))
        else:
            self.register_buffer("bias_buf", linear.bias.detach().contiguous())

        self.DIN = linear.in_features
        self.DOUT = linear.out_features

        self.BM, self.BN, self.BK = BM, BN, BK
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        orig_dtype = x.dtype
        orig_shape = x.shape

        # flatten if [B,S,D]
        if x.ndim == 3:
            x2 = x.reshape(-1, orig_shape[-1])
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        Bsz, DIN = x2.shape
        assert DIN == self.DIN, (DIN, self.DIN)

        # output fp32 then cast back
        y2 = torch.empty((Bsz, self.DOUT), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(Bsz, self.BM), triton.cdiv(self.DOUT, self.BN))
        MAX_K_TILES = triton.cdiv(DIN, self.BK)

        # Ensure bias buffer is on same device
        bias = self.bias_buf
        if bias.device != x.device:
            bias = bias.to(x.device)

        linear_baseline_kernel[grid](
            x2, self.weight, bias, y2,
            Bsz, DIN, self.DOUT,
            x2.stride(0), x2.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y2.stride(0), y2.stride(1),
            MAX_K_TILES=MAX_K_TILES,
            BLOCK_M=self.BM, BLOCK_N=self.BN, BLOCK_K=self.BK,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Cast back to keep the model consistent (fp16 recommended)
        y2 = y2.to(orig_dtype)

        if x.ndim == 3:
            return y2.reshape(orig_shape[0], orig_shape[1], self.DOUT)
        return y2


def replace_falcon_qkv_and_up_with_triton_baseline(model, BM=32, BN=64, BK=32, num_warps=4, num_stages=3):
    dev = next(model.parameters()).device
    counts = {"qkv": 0, "q": 0, "k": 0, "v": 0, "up": 0}

    for _, module in model.named_modules():
        # Falcon fused QKV
        if hasattr(module, "query_key_value") and isinstance(getattr(module, "query_key_value"), nn.Linear):
            module.query_key_value = TritonLinearBaseline(
                module.query_key_value, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages
            ).to(dev)
            counts["qkv"] += 1

        # Some variants expose separate q/k/v
        if hasattr(module, "q_proj") and isinstance(getattr(module, "q_proj"), nn.Linear):
            module.q_proj = TritonLinearBaseline(module.q_proj, BM=BM, BN=BN, BK=BK,
                                                 num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["q"] += 1
        if hasattr(module, "k_proj") and isinstance(getattr(module, "k_proj"), nn.Linear):
            module.k_proj = TritonLinearBaseline(module.k_proj, BM=BM, BN=BN, BK=BK,
                                                 num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["k"] += 1
        if hasattr(module, "v_proj") and isinstance(getattr(module, "v_proj"), nn.Linear):
            module.v_proj = TritonLinearBaseline(module.v_proj, BM=BM, BN=BN, BK=BK,
                                                 num_warps=num_warps, num_stages=num_stages).to(dev)
            counts["v"] += 1

        # Falcon MLP up projection
        if hasattr(module, "dense_h_to_4h") and isinstance(getattr(module, "dense_h_to_4h"), nn.Linear):
            module.dense_h_to_4h = TritonLinearBaseline(
                module.dense_h_to_4h, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages
            ).to(dev)
            counts["up"] += 1
        elif hasattr(module, "up_proj") and isinstance(getattr(module, "up_proj"), nn.Linear):
            module.up_proj = TritonLinearBaseline(
                module.up_proj, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages
            ).to(dev)
            counts["up"] += 1

    print("Replaced Falcon projections with TritonLinearBaseline:", counts)
    return model

def main():
    device = "cuda"
    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    replace_falcon_qkv_and_up_with_triton_baseline(model, BN=128, BK=32)
    # reuse your eval_piqa(dataset=valid_dataset) (first 10)
    acc = eval_piqa(model, tokenizer, valid_dataset)
    torch.cuda.synchronize()
    print("PIQA acc (first 10):", acc)

if __name__ == "__main__":
    main()