import os
import time
import csv
from tqdm import tqdm

# =====================
# CONFIG
# =====================
N_START = 4
N_END = 15          # included

# Daha açıklayıcı klasör
OUTPUT_DIR = "solution_space/raw_nqueens_solutions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# SYMMETRY UTILITIES
# =====================
def rot90(src, N):
    dst = [0] * N
    for c in range(N):
        dst[src[c]] = N - 1 - c
    return dst

def mirror(src, N):
    dst = [0] * N
    for c in range(N):
        dst[N - 1 - c] = src[c]
    return dst

def canonical(src, N):
    t = [None] * 8
    t[0] = src[:]
    t[1] = rot90(t[0], N)
    t[2] = rot90(t[1], N)
    t[3] = rot90(t[2], N)
    t[4] = mirror(t[0], N)
    t[5] = rot90(t[4], N)
    t[6] = rot90(t[5], N)
    t[7] = rot90(t[6], N)
    return min(t)

# =====================
# N-QUEENS SOLVER
# =====================
def is_safe(pos, col, row):
    for i in range(col):
        if pos[i] == row or \
           pos[i] - i == row - col or \
           pos[i] + i == row + col:
            return False
    return True

def solve_nqueen(N):
    pos = [0] * N
    uniq = []

    def backtrack(col):
        if col == N:
            canon = canonical(pos, N)
            if canon not in uniq:
                uniq.append(canon)
            return
        for r in range(N):
            if is_safe(pos, col, r):
                pos[col] = r
                backtrack(col + 1)

    backtrack(0)
    return uniq

# =====================
# MAIN LOOP
# =====================
print("\n🧩 N-Queens Solution Space Generation Started\n")

for N in range(N_START, N_END + 1):
    print(f"▶ Generating solution space for N = {N}")
    start = time.time()

    solutions = solve_nqueen(N)

    elapsed = time.time() - start
    print(f"  ✔ Unique (non-symmetric) solutions: {len(solutions)}")
    print(f"  ⏱ Time elapsed: {elapsed:.2f} seconds")

    # SAVE CSV
    file_path = os.path.join(
        OUTPUT_DIR,
        f"N{N}_raw_solution_space.csv"
    )

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"col_{i}" for i in range(N)])
        for sol in solutions:
            writer.writerow(sol)

    print(f"  💾 Saved to: {file_path}\n")

print("✅ All raw N-Queens solution spaces generated successfully.")