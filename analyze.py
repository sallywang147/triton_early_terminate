import re

def parse_dram_totals(path):
    total_read = 0.0
    total_write = 0.0

    # Regex captures things like:
    # dram__bytes_read.sum      Kbyte        29.57
    pattern = re.compile(
        r"(dram__bytes_read\.sum|dram__bytes_write\.sum)\s+\w+\s+([\d\.]+)",
        re.IGNORECASE
    )
    unit_pattern = re.compile(r"(Kbyte|Mbyte|Gbyte|byte)", re.IGNORECASE)

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                metric = match.group(1).lower()
                value = float(match.group(2))

                # Detect units
                unit_match = unit_pattern.search(line)
                unit = unit_match.group(1).lower() if unit_match else "byte"

                # Normalize to bytes
                if unit == "kbyte":
                    value *= 1024
                elif unit == "mbyte":
                    value *= 1024**2
                elif unit == "gbyte":
                    value *= 1024**3

                if "read" in metric:
                    total_read += value
                elif "write" in metric:
                    total_write += value

    return total_read, total_write


# ==== Paths to your uploaded files ====
baseline_file = "/mnt/data/triton_opt_trace.txt"        # :contentReference[oaicite:0]{index=0}
etu_file      = "/mnt/data/early_terminate_triton_opt_trace.txt"  # :contentReference[oaicite:1]{index=1}

# ==== Parse results ====
baseline_read, baseline_write = parse_dram_totals(baseline_file)
etu_read, etu_write = parse_dram_totals(etu_file)

# ==== Print final summary ====
def fmt(x): return f"{x/1024/1024:.3f} MB"

print("\n===== DRAM TOTALS =====")
print(f"Baseline DRAM Read : {fmt(baseline_read)}")
print(f"Baseline DRAM Write: {fmt(baseline_write)}\n")

print(f"ETU DRAM Read      : {fmt(etu_read)}")
print(f"ETU DRAM Write     : {fmt(etu_write)}\n")

print("===== REDUCTION =====")
print(f"Read reduction:  {(1 - etu_read / baseline_read) * 100:.2f}%")
print(f"Write reduction: {(1 - etu_write / baseline_write) * 100:.2f}%")
