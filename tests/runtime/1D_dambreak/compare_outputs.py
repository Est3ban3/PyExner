import numpy as np
import pnetcdf
from mpi4py import MPI


def load(fname):
    f = pnetcdf.File(fname, mode="r", comm=MPI.COMM_SELF)
    out = {v: np.array(f.variables[v][:]) for v in ("h", "hu", "hv")}
    f.close()
    return out


ref = load("output_ref.nc")
mpi = load("output_mpi.nc")

print(f"{'var':>4} {'snapshot':>8} {'max|diff|':>12} {'rel L2':>12}")
ok = True
for v in ("h", "hu", "hv"):
    a, b = ref[v], mpi[v]
    assert a.shape == b.shape, f"{v}: shapes differ {a.shape} vs {b.shape}"
    for t in range(a.shape[0]):
        d = np.abs(a[t] - b[t])
        denom = np.linalg.norm(a[t]) or 1.0
        rel = np.linalg.norm(a[t] - b[t]) / denom
        print(f"{v:>4} {t:>8} {d.max():>12.3e} {rel:>12.3e}")
        if d.max() > 1e-5:
            ok = False

print("\nVALIDATION:", "OK — multi-rank matches single-rank" if ok else "FAIL — fields differ")
