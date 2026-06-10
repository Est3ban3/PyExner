import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pnetcdf
from mpi4py import MPI


def make_gif(fname="output_gif.nc", out_gif="dambreak.gif", fps=12):
    f = pnetcdf.File(fname, mode="r", comm=MPI.COMM_SELF)
    x = np.array(f.variables["x"][:])
    y = np.array(f.variables["y"][:])
    h = np.array(f.variables["h"][:])  # (t, y, x)
    f.close()

    nt = h.shape[0]
    extent = [x.min(), x.max(), y.min(), y.max()]
    vmin, vmax = float(np.nanmin(h)), float(np.nanmax(h))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(h[0], origin="lower", extent=extent, aspect="equal",
                   cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, shrink=0.85, label="h [m]")
    title = ax.set_title("")

    def update(i):
        im.set_data(h[i])
        title.set_text(f"Altura de agua h  —  frame {i+1}/{nt}")
        return im, title

    anim = FuncAnimation(fig, update, frames=nt, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    print(f"Saved {out_gif}  ({nt} frames, grid {h.shape[2]}x{h.shape[1]}, "
          f"h in [{vmin:.3f}, {vmax:.3f}])")


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "output_gif.nc"
    out_gif = sys.argv[2] if len(sys.argv) > 2 else "dambreak.gif"
    make_gif(fname, out_gif)
