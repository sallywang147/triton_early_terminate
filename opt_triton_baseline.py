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
    Works with 2D [N, DIN] and 3D [B, S, DIN] by flattening.
    Returns output in the same dtype as input (fp16/bf16) to keep OPT happy.
    """
    def __init__(self, linear: nn.Linear, BM=32, BN=64, BK=32, num_warps=4, num_stages=3):
        super().__init__()
        assert linear.bias is not None, "OPT fc1 has bias"
        self.weight = nn.Parameter(linear.weight.detach().contiguous(), requires_grad=False)  # [DOUT, DIN]
        self.bias   = nn.Parameter(linear.bias.detach().contiguous(), requires_grad=False)    # [DOUT]
        self.DIN = linear.in_features
        self.DOUT = linear.out_features

        self.BM, self.BN, self.BK = BM, BN, BK
        self.num_warps = num_warps
        self.num_stages = num_stages

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

        # compute in fp32 output buffer
        y2 = torch.empty((Bsz, self.DOUT), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(Bsz, self.BM), triton.cdiv(self.DOUT, self.BN))
        MAX_K_TILES = triton.cdiv(DIN, self.BK)

        linear_baseline_kernel[grid](
            x2, self.weight, self.bias, y2,
            Bsz, DIN, self.DOUT,
            x2.stride(0), x2.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y2.stride(0), y2.stride(1),
            MAX_K_TILES=MAX_K_TILES,
            BLOCK_M=self.BM, BLOCK_N=self.BN, BLOCK_K=self.BK,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # IMPORTANT: cast back so fc2 (fp16) doesn't crash
        y2 = y2.to(orig_dtype)

        if x.ndim == 3:
            return y2.reshape(orig_shape[0], orig_shape[1], self.DOUT)
        return y2


def replace_opt_fc1_with_triton(model, BM=32, BN=64, BK=32, num_warps=4, num_stages=3):
    dev = next(model.parameters()).device
    replaced = 0
    for _, module in model.named_modules():
        if hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear):
            module.fc1 = TritonLinearBaseline(
                module.fc1,
                BM=BM, BN=BN, BK=BK,
                num_warps=num_warps,
                num_stages=num_stages,
            ).to(dev)
            torch.cuda.synchronize()
            replaced += 1
    print(f"Replaced {replaced} OPT MLP fc1 layers with TritonLinearBaseline")
    return model


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    device = "cuda"

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    # Patch FC1 in every decoder layer MLP
    replace_opt_fc1_with_triton(model, BN=128, BK=32)

    # Evaluate PIQA (validation is 1838 examples; test is 3084) :contentReference[oaicite:5]{index=5}
    acc = eval_piqa(model, tokenizer, valid_dataset)
    print("PIQA validation acc (limit=200):", acc)

if __name__ == "__main__":
    main()