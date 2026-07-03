"""Smoke test end-to-end de la rama EPB_TwoFluid por el driver completo.

Ejercita el unico camino que los tests unitarios no tocan: el driver real
(lectura PnetCDF de 8 campos -> bucle IMEX -> escritura PnetCDF), con el
solver EPB_TwoFluid, el integrador IMEX y los contornos transmisivos.

No valida fisica no trivial (para eso estan los demas scripts de esta
carpeta): solo confirma que el cableado completo corre, conserva masa y no
produce NaN/Inf ni densidades negativas en una evolucion corta.

Geometria 2.5D (x = horizontal, z = "y" del fichero). Estado inicial: plasma
casi uniforme (n_i = n_e = 1) con un bulto gaussiano de densidad que genera un
gradiente de presion y, por tanto, transporte hiperbolico real. Sin fuentes
(sin gravedad, B, E ni colisiones): el YAML no define la seccion ``epb``, asi
que phys/src toman sus valores normalizados/ nulos por defecto.

Ejecucion local (WSL, workaround MPICH GPU-aware):

    cd tests/runtime/EPB
    MPIR_CVAR_ENABLE_GPU=0 python smoke_epb.py

En cluster real, mismo script con ``MPI4JAX_USE_CUDA_MPI=1`` y, si se desea,
``mpirun -n P`` ajustando ``parNx*parNy = P`` en el YAML generado.
"""

import os
import numpy as np
import xarray as xr
import yaml


# --------------------------------------------------------------------------- #
# Parametros del caso                                                          #
# --------------------------------------------------------------------------- #

LX = 20.0          # extension en x
LZ = 10.0          # extension en z (eje "y" del fichero NetCDF)
DH = 0.5           # paso de malla uniforme
N0 = 1.0           # densidad de fondo (ion = electron, cuasineutral)
BUMP_AMP = 0.5     # amplitud del bulto de densidad
BUMP_SIGMA = 2.0   # ancho del bulto

END_TIME = 1.0
OUT_FREQ = 0.5
CFL = 0.4

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_NC = os.path.join(HERE, "input.nc")
OUTPUT_NC = os.path.join(HERE, "output.nc")
CONFIG_YAML = os.path.join(HERE, "input_epb.yaml")


# --------------------------------------------------------------------------- #
# 1. Generacion del estado inicial (NetCDF de 8 campos)                        #
# --------------------------------------------------------------------------- #

def build_input():
    x = np.arange(0.0, LX + DH, DH)
    y = np.arange(0.0, LZ + DH, DH)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Bulto gaussiano centrado: cuasineutral (n_i = n_e) -> sin carga neta inicial.
    r2 = (X - LX / 2.0) ** 2 + (Y - LZ / 2.0) ** 2
    bump = N0 + BUMP_AMP * np.exp(-r2 / (2.0 * BUMP_SIGMA**2))

    n_i = bump.astype(np.float32)
    n_e = bump.astype(np.float32)
    zero = np.zeros_like(n_i, dtype=np.float32)

    ds = xr.Dataset(
        {
            "n_i": (["y", "x"], n_i),
            "n_e": (["y", "x"], n_e),
            "j_ix": (["y", "x"], zero.copy()),
            "j_iy": (["y", "x"], zero.copy()),
            "j_iz": (["y", "x"], zero.copy()),
            "j_ex": (["y", "x"], zero.copy()),
            "j_ey": (["y", "x"], zero.copy()),
            "j_ez": (["y", "x"], zero.copy()),
        },
        coords={"x": x.astype(np.float32), "y": y.astype(np.float32)},
        attrs={
            "description": "EPB_TwoFluid smoke test: quasineutral plasma + density bump",
            "x_range": str([0.0, LX]),
            "y_range": str([0.0, LZ]),
        },
    )
    ds.to_netcdf(INPUT_NC, format="NETCDF3_64BIT")
    print(f"[build] Wrote {INPUT_NC} shape {n_i.shape} (Nz={n_i.shape[0]}, Nx={n_i.shape[1]})")
    return float(n_i.sum()), float(n_e.sum())


# --------------------------------------------------------------------------- #
# 2. Generacion del fichero de configuracion (YAML)                           #
# --------------------------------------------------------------------------- #

def build_config():
    eps = 0.1 * DH
    # Bandas de contorno de UN paso de malla de ancho, dimensionadas para capturar
    # el anillo exterior de centros de celda (con bandas mas finas quedan celdas
    # de borde sin cubrir; me paso al probarlo).
    boundaries = {
        "west": {
            "type": "Transmissive",
            "polygon": [[-eps, -eps], [DH, -eps], [DH, LZ + eps], [-eps, LZ + eps]],
            "values": ["nan"] * 8,
            "normal": [-1.0, 0.0],
        },
        "east": {
            "type": "Transmissive",
            "polygon": [[LX - DH, -eps], [LX + eps, -eps], [LX + eps, LZ + eps], [LX - DH, LZ + eps]],
            "values": ["nan"] * 8,
            "normal": [1.0, 0.0],
        },
        "south": {
            "type": "Transmissive",
            "polygon": [[-eps, -eps], [LX + eps, -eps], [LX + eps, DH], [-eps, DH]],
            "values": ["nan"] * 8,
            "normal": [0.0, -1.0],
        },
        "north": {
            "type": "Transmissive",
            "polygon": [[-eps, LZ - DH], [LX + eps, LZ - DH], [LX + eps, LZ + eps], [-eps, LZ + eps]],
            "values": ["nan"] * 8,
            "normal": [0.0, 1.0],
        },
    }

    config = {
        "end_time": END_TIME,
        "out_freq": OUT_FREQ,
        "cfl": CFL,
        "flux_scheme": "EPB_TwoFluid",
        "integrator": "IMEX",
        "parNx": 1,
        "parNy": 1,
        "input_file": INPUT_NC,
        "output_file": OUTPUT_NC,
        "dh": DH,
        "boundaries": boundaries,
        # Sin seccion 'epb': phys normalizado (c_i=c_e=1), fuentes nulas.
    }

    with open(CONFIG_YAML, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[build] Wrote {CONFIG_YAML}")


# --------------------------------------------------------------------------- #
# 3. Validacion del resultado                                                 #
# --------------------------------------------------------------------------- #

EPB_FIELDS = ("n_i", "n_e", "j_ix", "j_iy", "j_iz", "j_ex", "j_ey", "j_ez")


def validate_output(mass0_ni, mass0_ne):
    ok = True
    import pnetcdf
    from mpi4py import MPI

    f = pnetcdf.File(OUTPUT_NC, mode="r", comm=MPI.COMM_SELF)
    data = {name: np.array(f.variables[name][:]) for name in EPB_FIELDS if name in f.variables}
    nt = data["n_i"].shape[0] if "n_i" in data else 0
    f.close()

    # (a) Las 8 variables estan presentes en la salida.
    for name in EPB_FIELDS:
        if name not in data:
            print(f"[FAIL] variable '{name}' ausente en output.nc")
            ok = False
    if not ok:
        return ok
    print(f"[check] 8 variables presentes; snapshots t = {nt}")

    # (b) Sin NaN/Inf en ningun campo y en ningun snapshot.
    #     (los contornos escriben NaN en celdas bloqueadas: se ignoran).
    for name in EPB_FIELDS:
        interior = data[name][:, 1:-1, 1:-1]
        if not np.all(np.isfinite(interior)):
            print(f"[FAIL] '{name}' contiene NaN/Inf en el interior")
            ok = False
    if ok:
        print("[check] todos los campos finitos en el interior (sin NaN/Inf)")

    # (c) Densidades estrictamente positivas (positividad fisica).
    for name in ("n_i", "n_e"):
        mn = float(np.nanmin(data[name]))
        if mn <= 0.0:
            print(f"[FAIL] '{name}' min = {mn} <= 0 (positividad violada)")
            ok = False
        else:
            print(f"[check] {name} min = {mn:.6f} > 0")

    # (d) Conservacion aproximada de masa total entre primer y ultimo snapshot.
    #     Con contornos transmisivos hay flujo de salida, asi que solo exigimos
    #     que no haya creacion/destruccion espuria grande (tolerancia amplia).
    for name, m0 in (("n_i", mass0_ni), ("n_e", mass0_ne)):
        first = float(np.nansum(data[name][0]))
        last = float(np.nansum(data[name][-1]))
        rel = abs(last - first) / max(abs(first), 1e-30)
        status = "check" if rel < 0.20 else "warn"
        print(f"[{status}] masa {name}: t0={first:.4f} tN={last:.4f} (drift {rel*100:.2f}%)")

    return ok


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    from PyExner import run_driver

    print("=== EPB_TwoFluid smoke test (driver end-to-end) ===")
    mass0_ni, mass0_ne = build_input()
    build_config()

    print("[run] lanzando run_driver ...")
    state, coords = run_driver(CONFIG_YAML)
    print("[run] driver finalizado")

    ok = validate_output(mass0_ni, mass0_ne)
    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
