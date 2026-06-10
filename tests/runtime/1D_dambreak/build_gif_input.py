import sys
import numpy as np
import xarray as xr


def build_circular_dambreak(N=300, L=100.0, hin=1.0, hout=0.2, R=20.0,
                            fname="input_gif.nc"):
    """Circular dam-break: high water column of radius R collapses radially."""
    dh = L / (N - 1)
    x = np.linspace(0.0, L, N)
    y = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x, y, indexing="xy")

    r = np.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)
    h = np.where(r <= R, hin, hout)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)
    n = np.zeros_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h.astype(np.float32)),
            "hu": (["y", "x"], (h * u).astype(np.float32)),
            "hv": (["y", "x"], (h * v).astype(np.float32)),
            "z": (["y", "x"], z.astype(np.float32)),
            "n": (["y", "x"], n.astype(np.float32)),
        },
        coords={"x": x.astype(np.float32), "y": y.astype(np.float32)},
        attrs={"description": f"{N}x{N} circular dam-break", "dh": str(dh)},
    )
    ds.to_netcdf(fname, format="NETCDF3_64BIT")
    print(f"Wrote {fname}: {N}x{N} = {N*N:,} cells, dh={dh:.5f}, R={R}")
    return dh


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    build_circular_dambreak(N=N)
