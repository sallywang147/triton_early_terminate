import triton
import torch
import tqdm
import triton.language as tl
from datasets import load_dataset
from transformers import OPTForCausalLM, AutoTokenizer
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

        # ✅ acc stays 1D: [BLOCK_N]
        acc += partial * still_active.to(tl.float32)

    # Add bias
    out = tl.where(passed, 0.0, acc)
    tl.store(
        Y_ptr + pid_m * stride_ym + offs_n * stride_yn,
        out,
        mask=n_mask,
    )



class TritonLinearETRow(nn.Module):
    """
    Calls the user's unmodified linear_etu_kernel.
    Works for OPT fc1 on inputs [B,S,D] or [N,D] by flattening to [N,D].
    Casts output back to input dtype to keep OPT happy.
    """
    def __init__(self, linear: nn.Linear, BN=64, BK=32, BM=1, num_warps=4, num_stages=3):
        super().__init__()
        assert BN <= 64, "Kernel has mask_n = offs_n < 64; BN must be <= 64."
        self.weight = nn.Parameter(linear.weight.detach().contiguous(), requires_grad=False)
        # OPT fc1 has bias, but your kernel does not use B_ptr. We'll keep it for signature compat.
        self.bias = nn.Parameter(linear.bias.detach().contiguous(), requires_grad=False) if linear.bias is not None else None

        self.DIN = linear.in_features
        self.DOUT = linear.out_features

        self.BM = BM  # unused by kernel math, but required by signature
        self.BN = BN
        self.BK = BK
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

        # kernel writes fp32 (as it accumulates in fp32)
        y2 = torch.empty((Bsz, self.DOUT), device=x.device, dtype=torch.float32)

        grid = (Bsz, triton.cdiv(self.DOUT, self.BN))

        # pass B_ptr even though kernel doesn't add bias
        B_ptr = self.bias if self.bias is not None else torch.zeros((self.DOUT,), device=x.device, dtype=self.weight.dtype)

        linear_etu_kernel[grid](
            x2, self.weight, B_ptr, y2,
            Bsz, DIN, self.DOUT,
            x2.stride(0), x2.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y2.stride(0), y2.stride(1),
            BLOCK_M=self.BM,
            BLOCK_N=self.BN,
            BLOCK_K=self.BK,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        y2 = y2.to(orig_dtype)
        if x.ndim == 3:
            return y2.reshape(orig_shape[0], orig_shape[1], self.DOUT)
        return y2


def replace_opt_fc1_with_linear_etu_kernel(model, BN=64, BK=32, BM=1, num_warps=4, num_stages=3):
    dev = next(model.parameters()).device
    replaced = 0
    for _, module in model.named_modules():
        if hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear):
            module.fc1 = TritonLinearETRow(
                module.fc1,
                BN=BN, BK=BK, BM=BM,
                num_warps=num_warps,
                num_stages=num_stages,
            ).to(dev)
            replaced += 1
    print(f"Replaced {replaced} OPT fc1 layers with TritonLinearET_UnmodifiedKernel")
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
    replace_opt_fc1_with_linear_etu_kernel(model, BN=64, BK=32, BM=1)

    # Evaluate PIQA (validation is 1838 examples; test is 3084) :contentReference[oaicite:5]{index=5}
    acc = eval_piqa(model, tokenizer, valid_dataset)
    print("PIQA validation acc (limit=200):", acc)

if __name__ == "__main__":
    main()