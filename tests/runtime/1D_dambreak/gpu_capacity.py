import os
import sys
import gc
import numpy as np

# Real allocation (no 75% preallocation) so we can read true peak usage.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import xarray as xr

from PyExner.runtime.driver import run_driver


def build_input(N, fname, L=100.0):
    x = np.linspace(0.0, L, N)
    y = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)
    h = np.where(r <= 20.0, 1.0, 0.2)
    z = np.ones_like(h)
    n = np.zeros_like(h)
    ds = xr.Dataset(
        {
            "h": (["y", "x"], h.astype(np.float32)),
            "hu": (["y", "x"], np.zeros_like(h, np.float32)),
            "hv": (["y", "x"], np.zeros_like(h, np.float32)),
            "z": (["y", "x"], z.astype(np.float32)),
            "n": (["y", "x"], n.astype(np.float32)),
        },
        coords={"x": x.astype(np.float32), "y": y.astype(np.float32)},
    )
    ds.to_netcdf(fname, format="NETCDF3_64BIT")


YAML_TMPL = """end_time: {end}
out_freq: 1000.0
cfl: 0.5
flux_scheme: Roe
integrator: Forward Euler
parNx: 1
parNy: 1
input_file: {inp}
output_file: {out}
dh: {dh}

boundaries:
  inlet:
    type: Transmissive
    polygon: [[-0.1,-0.1],[0.1,-0.1],[0.1,100.1],[-0.1,100.1]]
    values: [0.0,0.0,0.0,0.0]
    normal: [-1.0,0.0]
  outlet:
    type: Transmissive
    polygon: [[99.9,-0.1],[100.1,-0.1],[100.1,100.1],[99.9,100.1]]
    values: [0.0,0.0,0.0,0.0]
    normal: [1.0,0.0]
  bottom:
    type: Transmissive
    polygon: [[-0.1,-0.1],[100.1,-0.1],[100.1,0.1],[-0.1,0.1]]
    values: [0.0,0.0,0.0,0.0]
    normal: [0.0,-1.0]
  top:
    type: Transmissive
    polygon: [[-0.1,99.9],[100.1,99.9],[100.1,100.1],[-0.1,100.1]]
    values: [0.0,0.0,0.0,0.0]
    normal: [0.0,1.0]
"""


def measure(N):
    inp = f"_probe_{N}.nc"
    out = f"_probe_out_{N}.nc"
    yml = f"_probe_{N}.yaml"
    build_input(N, inp)
    dh = 100.0 / (N - 1)
    # tiny end_time -> just a few steps
    with open(yml, "w") as f:
        f.write(YAML_TMPL.format(end=dh * 0.1, inp=inp, out=out, dh=dh))

    dev = jax.devices()[0]
    try:
        dev.memory_stats()  # reset baseline reference
    except Exception:
        pass

    run_driver(yml)
    jax.block_until_ready(jnp.zeros(1))
    stats = dev.memory_stats()
    peak = stats.get("peak_bytes_in_use", 0)
    cur = stats.get("bytes_in_use", 0)
    for fn in (inp, out, yml):
        try:
            os.remove(fn)
        except OSError:
            pass
    gc.collect()
    return peak, cur


if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1:]] or [1000, 2000, 3000, 4000]
    print(f"{'N':>7} {'cells':>14} {'peak GB':>10} {'B/cell':>10}")
    for N in Ns:
        try:
            peak, cur = measure(N)
            cells = N * N
            print(f"{N:>7} {cells:>14,} {peak/1e9:>10.3f} {peak/cells:>10.1f}")
        except Exception as e:
            print(f"{N:>7} FAILED: {type(e).__name__}: {str(e)[:80]}")
            break
