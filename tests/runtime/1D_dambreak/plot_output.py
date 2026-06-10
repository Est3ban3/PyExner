import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pnetcdf
from mpi4py import MPI


def plot_output(fname="output_large_gpu.nc", out_png="dambreak_field.png"):
    f = pnetcdf.File(fname, mode="r", comm=MPI.COMM_SELF)
    x = np.array(f.variables["x"][:])
    y = np.array(f.variables["y"][:])
    h = np.array(f.variables["h"][:])  # (t, y, x)
    f.close()

    nt = h.shape[0]
    t_first, t_last = 0, nt - 1

    extent = [x.min(), x.max(), y.min(), y.max()]
    vmin = float(np.nanmin(h))
    vmax = float(np.nanmax(h))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for ax, ti, label in zip(axes, [t_first, t_last], ["t = inicial", "t = final"]):
        im = ax.imshow(
            h[ti],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Altura de agua h  ({label})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.85, label="h [m]")

    fig.suptitle(f"Dam-break {h.shape[2]}x{h.shape[1]}  —  {fname}", fontsize=13)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}  (grid {h.shape[2]}x{h.shape[1]}, {nt} snapshots, "
          f"h in [{vmin:.3f}, {vmax:.3f}])")


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "output_large_gpu.nc"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "dambreak_field.png"
    plot_output(fname, out_png)
