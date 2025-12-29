import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =====================
# CONFIG
# =====================
# Handcrafted feature veya raw solution space değil:
# Autoencoder, ham solution space üzerinde çalışıyor
INPUT_DIR = "solution_space/raw_nqueens_solutions"

# Öğrenilmiş latent solution space
OUTPUT_DIR = "solution_space/learned_latent_space"

# ---- DYNAMIC N RANGE ----
N_START = 4
N_END   = 15   # inclusive

EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
LATENT_RATIO = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# UTILS
# =====================
def extract_N(filename):
    # N8_raw_solution_space.csv -> 8
    return int(filename.split("_")[0][1:])

# =====================
# AUTOENCODER MODEL
# =====================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# =====================
# TRAIN FUNCTION
# =====================
def train_autoencoder(X, N):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    latent_dim = max(2, int(N * LATENT_RATIO))
    model = AutoEncoder(N, latent_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    epoch_bar = tqdm(
        range(EPOCHS),
        desc=f"🧠 Training Autoencoder (N={N})",
        leave=True
    )

    for epoch in epoch_bar:
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(DEVICE)

            recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_bar.set_postfix(loss=f"{total_loss:.4f}")

    return model

# =====================
# MAIN
# =====================
print("\n🤖 AUTOENCODER LEARNING ON SOLUTION SPACE STARTED\n")

csv_files = sorted(
    [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith("_raw_solution_space.csv")
        and N_START <= extract_N(f) <= N_END
    ],
    key=extract_N
)

file_bar = tqdm(csv_files, desc="📂 Processing solution spaces")

for file in file_bar:
    path = os.path.join(INPUT_DIR, file)

    N = extract_N(file)
    file_bar.set_postfix(N=N)

    # ---------------------
    # LOAD SOLUTION SPACE
    # ---------------------
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        solutions = []
        for row in reader:
            sol = list(map(int, row))
            solutions.append(sol)

    X = np.array(solutions)

    # Safety check
    assert X.shape[1] == N, (
        f"Column mismatch in {file}: expected {N}, got {X.shape[1]}"
    )

    # ---------------------
    # TRAIN AUTOENCODER
    # ---------------------
    model = train_autoencoder(X, N)
    model.eval()

    # ---------------------
    # EXTRACT LATENT SPACE
    # ---------------------
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        latent_chunks = []
        latent_bar = tqdm(
            range(0, len(X), BATCH_SIZE),
            desc=f"🔎 Extracting Latent Space (N={N})",
            leave=False
        )

        for i in latent_bar:
            batch = X_tensor[i:i + BATCH_SIZE]
            _, latent = model(batch)
            latent_chunks.append(latent.cpu().numpy())

    latent = np.vstack(latent_chunks)

    # ---------------------
    # SAVE LATENT SPACE
    # ---------------------
    out_file = os.path.join(
        OUTPUT_DIR,
        f"N{N}_learned_latent_space.csv"
    )

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"z{i}" for i in range(latent.shape[1])])
        writer.writerows(latent)

    tqdm.write(f"💾 Saved latent space: {out_file}")

print("\n✅ AUTOENCODER LATENT SPACE GENERATION COMPLETED")
print(f"📁 Output directory: {OUTPUT_DIR}")