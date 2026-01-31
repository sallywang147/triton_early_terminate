import re

def sum_linear_fwd_elapsed_cycles(txt_path: str) -> int:
    """
    Sum all 'Elapsed Cycles' entries that belong to linear_fwd_kernel blocks.
    """
    with open(txt_path, "r") as f:
        text = f.read()

    # Regex explanation:
    # 1. Match a linear_fwd_kernel header
    # 2. Non-greedily consume everything until
    # 3. Capture the Elapsed Cycles number
    pattern = re.compile(
        r"linear_fwd_kernel[^\n]*\n"          # kernel header
        r"(?:.*\n)*?"                         # kernel body (non-greedy)
        r"\s*Elapsed Cycles\s+cycle\s+([\d,]+)",
        re.MULTILINE
    )

    total_cycles = 0
    matches = pattern.findall(text)

    for val in matches:
        total_cycles += int(val.replace(",", ""))

    return total_cycles

def sum_overhead_cycles(txt_path: str) -> int:
    """
    Sum all 'Elapsed Cycles' entries that belong to linear_fwd_kernel blocks.
    """
    with open(txt_path, "r") as f:
        text = f.read()

    # Regex explanation:
    # 1. Match a linear_fwd_kernel header
    # 2. Non-greedily consume everything until
    # 3. Capture the Elapsed Cycles number
    pattern = re.compile(
        r"prefix_y1_cumsq_stats_kernel_fast[^\n]*\n"          # kernel header
        r"(?:.*\n)*?"                         # kernel body (non-greedy)
        r"\s*Elapsed Cycles\s+cycle\s+([\d,]+)",
        re.MULTILINE
    )

    total_cycles = 0
    matches = pattern.findall(text)

    for val in matches:
        total_cycles += int(val.replace(",", ""))

    return total_cycles


if __name__ == "__main__":
    path = "opt_cycles_update.txt"  # <-- change to your file
    path2 = "early_opt_cycles_update.txt"  # <-- change to your file
    total = sum_linear_fwd_elapsed_cycles(path)
    total2 = sum_linear_fwd_elapsed_cycles(path2)
    overhead = sum_overhead_cycles(path2)
    print(f"Total Elapsed Cycles (linear_fwd_kernel) without early termination: {total:,}")
    print(f"Total Elapsed Cycles (linear_fwd_kernel) with early termination: {total2:,}")
    print(f"Total Overhead Cycles (prefix_y1_cumsq_stats_kernel_fast): {overhead:,}")
    print(f"Reduction factoring in overhead: {total - total2 - overhead:,} cycles ({(total - total2 - overhead) / total * 100:.2f}%)")
    print(f"Reduction without overhead processing: {total - total2:,} cycles ({(total - total2) / total * 100:.2f}%)")