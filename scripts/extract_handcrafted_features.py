import os
import csv
import time
import numpy as np
from tqdm import tqdm

# =====================
# CONFIG
# =====================
# Ham çözüm uzayı
INPUT_DIR = "solution_space/raw_nqueens_solutions"

# İnsan tanımlı feature uzayı
OUTPUT_DIR = "solution_space/handcrafted_feature_space"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(sol):
    sol = np.array(sol)
    N = len(sol)

    diffs = np.diff(sol)

    features = {
        # --- META ---
        "N": N,

        # --- BASIC STATISTICS ---
        "row_mean": sol.mean(),
        "row_std": sol.std(),
        "row_min": sol.min(),
        "row_max": sol.max(),
        "row_range": sol.max() - sol.min(),

        # --- CENTER DISTRIBUTION ---
        "center_dist_mean": np.mean(np.abs(sol - (N - 1) / 2)),
        "center_dist_std": np.std(np.abs(sol - (N - 1) / 2)),

        # --- ADJACENT COLUMN RELATIONS ---
        "adj_diff_mean": np.mean(np.abs(diffs)),
        "adj_diff_std": np.std(np.abs(diffs)),
        "adj_diff_max": np.max(np.abs(diffs)),
        "adj_diff_min": np.min(np.abs(diffs)),

        # --- GLOBAL STRUCTURE ---
        "increasing_pairs_ratio": np.mean(diffs > 0),
        "decreasing_pairs_ratio": np.mean(diffs < 0),
        "flat_pairs_ratio": np.mean(diffs == 0),

        # --- DISTRIBUTION SHAPE ---
        "row_skewness": (
            np.mean((sol - sol.mean()) ** 3) /
            (np.std(sol) ** 3 + 1e-9)
        ),
        "row_kurtosis": (
            np.mean((sol - sol.mean()) ** 4) /
            (np.std(sol) ** 4 + 1e-9)
        ),

        # --- SPACING ---
        "unique_rows_ratio": len(np.unique(sol)) / N,
        "even_row_ratio": np.mean(sol % 2 == 0),
        "odd_row_ratio": np.mean(sol % 2 == 1),

        # --- POSITIONAL ENERGY ---
        "positional_energy": np.sum(np.abs(sol - np.arange(N))),
        "diagonal_energy": np.sum(np.abs(sol - np.flip(np.arange(N))))
    }

    return features

# =====================
# MAIN ANALYSIS
# =====================
print("\n🔍 Handcrafted Feature Space Extraction Started\n")

summary = []

csv_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
)

for file in tqdm(csv_files, desc="Processing raw solution spaces"):
    file_path = os.path.join(INPUT_DIR, file)

    # N bilgisi: N8_raw_solution_space.csv → 8
    try:
        N = int(file.split("_")[0][1:])
    except:
        continue

    start = time.time()
    all_features = []

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # column headers

        for idx, row in enumerate(reader):
            sol = list(map(int, row))
            feats = extract_features(sol)
            feats["solution_id"] = idx
            all_features.append(feats)

    elapsed = time.time() - start

    # SAVE FEATURES
    out_file = os.path.join(
        OUTPUT_DIR,
        f"N{N}_handcrafted_feature_space.csv"
    )

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_features[0].keys())
        for row in all_features:
            writer.writerow(row.values())

    summary.append([N, len(all_features), f"{elapsed:.2f}"])

# =====================
# SAVE SUMMARY
# =====================
summary_path = os.path.join(
    OUTPUT_DIR,
    "summary_handcrafted_feature_space.csv"
)

with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "num_solutions", "analysis_time_sec"])
    writer.writerows(summary)

print("\n✅ Handcrafted feature extraction completed successfully.")
print(f"📁 Output directory: {OUTPUT_DIR}/")