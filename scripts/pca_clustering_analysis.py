import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# =====================
# CONFIG
# =====================
# Öğrenilmiş latent solution space
LATENT_DIR = "solution_space/learned_latent_space"

# Analiz çıktıları
PER_N_OUT_DIR = "analysis_outputs/per_N_solution_space_analysis"
SUMMARY_OUT_DIR = "analysis_outputs/global_structure_summary"

N_START = 4
N_END = 15

PCA_COMPONENTS = 2
KMEANS_K = 3

# Hybrid silhouette settings
FULL_SILHOUETTE_UP_TO_N = 10
MAX_SAMPLE_SIZE = 5000

os.makedirs(PER_N_OUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_OUT_DIR, exist_ok=True)

# =====================
# UTILS
# =====================
def extract_N(filename):
    # N8_learned_latent_space.csv -> 8
    return int(filename.split("_")[0][1:])

# =====================
# MAIN ANALYSIS
# =====================
print("\n📊 PCA + CLUSTERING ANALYSIS ON SOLUTION SPACE STARTED\n")

latent_files = sorted(
    [
        f for f in os.listdir(LATENT_DIR)
        if f.endswith("_learned_latent_space.csv")
        and N_START <= extract_N(f) <= N_END
    ],
    key=extract_N
)

summary_rows = []

overall_bar = tqdm(
    latent_files,
    desc="Processing N values",
    dynamic_ncols=True
)

try:
    for file in overall_bar:
        N = extract_N(file)
        overall_bar.set_postfix(N=N)

        out_n_dir = os.path.join(PER_N_OUT_DIR, f"N{N}")
        os.makedirs(out_n_dir, exist_ok=True)

        # -------- LOAD LATENT SPACE --------
        X = pd.read_csv(os.path.join(LATENT_DIR, file)).values
        n_samples, n_features = X.shape

        # -------- PCA CHECK --------
        if min(n_samples, n_features) < 2:
            tqdm.write(f"⚠️ Skipping N={N}: insufficient data")
            continue

        # -------- PCA --------
        pca = PCA(n_components=PCA_COMPONENTS)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()

        pd.DataFrame(
            X_pca,
            columns=["PC1", "PC2"]
        ).to_csv(
            os.path.join(out_n_dir, "pca_projection.csv"),
            index=False
        )

        # -------- CLUSTERING --------
        k = min(KMEANS_K, n_samples)
        if k < 2:
            tqdm.write(f"⚠️ Skipping clustering for N={N}")
            continue

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_pca)

        # -------- SILHOUETTE (HYBRID) --------
        sil_score = None
        sil_mode = "NA"

        if n_samples > 2:
            if N <= FULL_SILHOUETTE_UP_TO_N:
                sil_score = silhouette_score(X_pca, labels)
                sil_mode = "FULL"
            else:
                sample_size = min(MAX_SAMPLE_SIZE, n_samples)
                sil_score = silhouette_score(
                    X_pca,
                    labels,
                    sample_size=sample_size,
                    random_state=42
                )
                sil_mode = f"SAMPLED_{sample_size}"

        # -------- PCA PLOT --------
        plt.figure(figsize=(6, 6))
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            s=6,
            cmap="tab10",
            alpha=0.8
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA + KMeans Clustering (N={N})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_n_dir, "pca_clustering_plot.png"),
            dpi=200
        )
        plt.close()

        # -------- METRICS FILE --------
        with open(os.path.join(out_n_dir, "metrics.txt"), "w") as f:
            f.write(f"N = {N}\n")
            f.write(f"PCA explained variance (PC1+PC2): {explained:.6f}\n")
            f.write(f"KMeans k = {k}\n")
            if sil_score is not None:
                f.write(f"Silhouette ({sil_mode}): {sil_score:.6f}\n")
            else:
                f.write("Silhouette: not defined\n")

        summary_rows.append({
            "N": N,
            "pca_variance": explained,
            "silhouette": sil_score,
            "silhouette_mode": sil_mode
        })

        tqdm.write(
            f"N={N} | PCA={explained:.2f} | k={k} | "
            f"silhouette={sil_score:.4f} ({sil_mode})"
            if sil_score is not None else
            f"N={N} | PCA={explained:.2f} | k={k} | silhouette=NA"
        )

except KeyboardInterrupt:
    print("\n⛔ Interrupted by user.")

# =====================
# SUMMARY + GLOBAL PLOTS
# =====================
print("\n📈 GENERATING GLOBAL SUMMARY FILES & PLOTS\n")

summary_df = pd.DataFrame(summary_rows).sort_values("N")

summary_df.to_csv(
    os.path.join(SUMMARY_OUT_DIR, "summary_metrics.csv"),
    index=False
)

# ---- Silhouette vs N ----
plt.figure(figsize=(7, 4))
plt.plot(summary_df["N"], summary_df["silhouette"], marker="o")
plt.xlabel("N")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs N")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(SUMMARY_OUT_DIR, "silhouette_vs_N.png"),
    dpi=200
)
plt.close()

# ---- PCA variance vs N ----
plt.figure(figsize=(7, 4))
plt.plot(
    summary_df["N"],
    summary_df["pca_variance"],
    marker="o",
    color="orange"
)
plt.xlabel("N")
plt.ylabel("Explained Variance (PC1 + PC2)")
plt.title("PCA Explained Variance vs N")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(SUMMARY_OUT_DIR, "pca_variance_vs_N.png"),
    dpi=200
)
plt.close()

print("\n✅ STRUCTURAL ANALYSIS COMPLETED SUCCESSFULLY")
print(f"📁 Per-N analysis: {PER_N_OUT_DIR}")
print(f"📁 Global summary: {SUMMARY_OUT_DIR}")

# =====================
# SLEEP SYSTEM (macOS)
# =====================
#print("\n💤 Analysis finished. System going to sleep...")
#os.system("pmset sleepnow")