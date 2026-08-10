import time
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
print("N", cat.N, "Q", cat.Q)

# isolate matmul speed
B32 = np.full((cat.N, 6), 1.0/cat.N, dtype=np.float32)
XT = np.ascontiguousarray(cat.X.T)  # (Q,N) contiguous
t0 = time.perf_counter()
for _ in range(20):
    P = XT @ B32
print("20x XT@B32 (both f32, XT contig):", time.perf_counter()-t0, "s")

B64 = np.full((cat.N, 6), 1.0/cat.N, dtype=np.float64)
t0 = time.perf_counter()
for _ in range(20):
    P = cat.X.T @ B64  # transposed view, mixed dtype
print("20x X.T@B64 (mixed dtype, transposed view):", time.perf_counter()-t0, "s")

print("numpy config:")
np.show_config()
