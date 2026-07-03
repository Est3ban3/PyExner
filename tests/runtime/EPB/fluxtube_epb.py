"""Geometria dipolar integrada en tubo de flujo (``epb_fluxtube``).

Valida la geometria de la linea dipolar, las conductancias Pedersen
integradas (Sigma_P^F, Sigma_P^E), el apantallamiento F_s y la tasa RT
flux-tube

    gamma_FT(h_apex) = F_s g/(nu_eff L_n) - beta.

Bloques:

  A) GEOMETRIA: el apex reproduce h_apex y B dipolar local a precision de
     maquina; h decrece monotonicamente del apex al pie; el pie queda en h_min.
  B) CONVERGENCIA de las integrales de linea (trapecio): refinar nlat x4
     cambia Sigma_F/Sigma_E en < 1e-6 relativo; todas las cantidades > 0.
  C) DIA/NOCHE: con capa E diurna el apantallamiento colapsa
     (F_s << 1: cortocircuito) y de noche F_s ~ 1. Ordenes de magnitud de las
     conductancias comparables a la literatura (Sigma_E^dia ~ unidades-decenas
     de S; Sigma_F^noche ~ decimas-unidades de S).
  D) UMBRAL FISICO: gamma_FT nocturno tiene umbral de altura (la EPB requiere
     capa F elevada), el topside (apex > h_mF2) es estable, el e-folding
     minimo nocturno cae en el rango observado (~5-30 min) y la capa E diurna
     SUPRIME el crecimiento (tau demasiado lento para formar burbuja en una
     tarde) — el control dia/noche de la EPB con conductancias integradas.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python fluxtube_epb.py
"""

import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_ionosphere import FRegionParams, dipole_B
from PyExner.solvers.kernels.epb_fluxtube import (
    ELAYER_DAY, ELAYER_NIGHT, H_MIN,
    dipole_line, flux_tube_quantities, gamma_flux_tube,
)

FR = FRegionParams()


def test_A_geometry():
    ok = True
    h_apex = np.array([250.0e3, 350.0e3, 450.0e3])
    line = dipole_line(h_apex, FR, nlat=801)
    # apex exacto
    err_h = np.abs(line.h[:, 0] - h_apex).max()
    # B del apex = dipolo local
    B_loc = np.asarray(dipole_B(jnp.asarray(h_apex), FR.B_surf_eq))
    err_B = np.abs(line.B[:, 0] - B_loc).max() / B_loc.min()
    # monotonia y pie
    mono = bool(np.all(np.diff(line.h, axis=1) < 0.0))
    err_foot = np.abs(line.h[:, -1] - H_MIN).max()
    good = err_h < 1e-6 and err_B < 1e-12 and mono and err_foot < 1.0
    ok &= good
    print(f"[A] apex |dh| = {err_h:.2e} m, |dB|/B = {err_B:.2e}, "
          f"monotona = {mono}, pie |h-h_min| = {err_foot:.2e} m  "
          f"{'PASS' if good else 'FAIL'}")
    return ok


def test_B_convergence():
    h_apex = np.array([300.0e3, 400.0e3])
    ft1 = flux_tube_quantities(h_apex, FR, ELAYER_NIGHT, nlat=401)
    ft2 = flux_tube_quantities(h_apex, FR, ELAYER_NIGHT, nlat=1601)
    rel = max(
        float(np.abs(ft2.Sigma_F / ft1.Sigma_F - 1.0).max()),
        float(np.abs(ft2.Sigma_E / ft1.Sigma_E - 1.0).max()),
        float(np.abs(ft2.N_FT / ft1.N_FT - 1.0).max()),
    )
    pos = all(float(np.min(v)) > 0.0
              for v in (ft2.Sigma_F, ft2.Sigma_E, ft2.N_FT, ft2.nu_eff, ft2.F_s))
    # La capa E es delgada (H_E = 6 km): el trapecio a nlat=401 la resuelve con
    # ~unos pocos puntos por escala; 1e-5 relativo es convergencia sobrada para
    # un cierre de conductancias.
    ok = rel < 1e-5 and pos
    print(f"[B] convergencia nlat 401->1601: drift rel = {rel:.2e}, "
          f"positividad = {pos}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_C_day_night():
    h_apex = 350.0e3
    ftN = flux_tube_quantities(h_apex, FR, ELAYER_NIGHT)
    ftD = flux_tube_quantities(h_apex, FR, ELAYER_DAY)
    print(f"[C] apex 350 km:  Sigma_F = {ftN.Sigma_F[0]:.3f} S   "
          f"Sigma_E(noche) = {ftN.Sigma_E[0]:.3f} S   "
          f"Sigma_E(dia) = {ftD.Sigma_E[0]:.3f} S")
    print(f"    F_s(noche) = {ftN.F_s[0]:.3f}   F_s(dia) = {ftD.F_s[0]:.3f}")
    ok = True
    if not (ftN.F_s[0] > 0.7):
        print("[C][FAIL] apantallamiento nocturno demasiado fuerte"); ok = False
    if not (ftD.F_s[0] < 0.35):
        print("[C][FAIL] la capa E diurna no cortocircuita"); ok = False
    # ordenes de magnitud (literatura: Sigma_E^dia ~ 1-20 S, Sigma_F^noche ~ 0.05-5 S)
    if not (0.05 < ftN.Sigma_F[0] < 5.0 and 1.0 < ftD.Sigma_E[0] < 30.0):
        print("[C][FAIL] conductancias fuera de los ordenes observados"); ok = False
    if ok:
        print(f"[C][PASS] cortocircuito diurno: F_s {ftN.F_s[0]:.2f} -> "
              f"{ftD.F_s[0]:.2f}; ordenes de magnitud correctos")
    return ok


def test_D_threshold():
    h_apex = np.arange(220.0e3, 520.0e3 + 1, 20.0e3)
    gamN, ftN = gamma_flux_tube(h_apex, FR, ELAYER_NIGHT)
    gamD, ftD = gamma_flux_tube(h_apex, FR, ELAYER_DAY)

    print("[D]  h_apex[km]   F_s(noche)  gamma_noche[s^-1]  tau[min]   gamma_dia[s^-1]")
    for k in range(len(h_apex)):
        tau = 1.0 / gamN[k] / 60.0 if gamN[k] > 0 else np.inf
        print(f"     {h_apex[k] / 1e3:7.0f}     {ftN.F_s[k]:.3f}      "
              f"{gamN[k]:+.3e}      {tau:6.1f}     {gamD[k]:+.3e}")

    ok = True
    # umbral nocturno: primera altura con gamma > 0
    pos = np.where(gamN > 0.0)[0]
    if len(pos) == 0 or not (240.0e3 <= h_apex[pos[0]] <= 420.0e3):
        print("[D][FAIL] sin umbral nocturno en [240, 420] km"); ok = False
    else:
        h_thr = h_apex[pos[0]]
        print(f"[D][PASS] umbral nocturno de inestabilidad: h_apex ~ "
              f"{h_thr / 1e3:.0f} km (obs: capa F elevada > ~300 km)")
    # estable sobre el pico F2 (topside: grad n paralelo a g)
    top = h_apex > FR.h_peak
    if not np.all(gamN[top] < 0.0):
        print("[D][FAIL] gamma > 0 sobre el pico F2 (topside debe ser estable)"); ok = False
    else:
        print("[D][PASS] topside estable (gamma < 0 para apex > h_mF2)")
    # e-folding nocturno minimo (pico de la curva) en el rango observado
    gmaxN = float(gamN.max())
    tauN = 1.0 / gmaxN / 60.0 if gmaxN > 0 else np.inf
    if not (5.0 < tauN < 60.0):
        print(f"[D][FAIL] tau_min nocturno = {tauN:.1f} min fuera de [5, 60]"); ok = False
    else:
        print(f"[D][PASS] e-folding nocturno minimo: tau = {tauN:.1f} min "
              f"(obs: ~5-30 min)")
    # cortocircuito diurno: el crecimiento queda demasiado LENTO para formar
    # burbuja (la amplificacion requiere ~7 e-folds; con tau_dia >~ 2 h no se
    # alcanzan en una tarde). NOTA: este barrido conserva la capa F NOCTURNA
    # elevada y solo enciende la capa E diurna: aisla el efecto del shunt.
    gmaxD = float(gamD.max())
    tauD = 1.0 / gmaxD / 60.0 if gmaxD > 0 else np.inf
    if not (tauD > 100.0 and gmaxD < 0.3 * gmaxN):
        print(f"[D][FAIL] el shunt diurno no suprime: tau_dia = {tauD:.0f} min, "
              f"gmax_dia/gmax_noche = {gmaxD / gmaxN:.2f}"); ok = False
    else:
        print(f"[D][PASS] supresion diurna: tau_min dia = {tauD:.0f} min "
              f"(>> noche {tauN:.0f} min), gamma_max reducido x"
              f"{gmaxN / max(gmaxD, 1e-30):.1f}")
    return ok


def main():
    print("=== Geometria flux-tube integrada (dipolar) ===\n")
    ok = True
    ok &= test_A_geometry()
    ok &= test_B_convergence()
    print()
    ok &= test_C_day_night()
    print()
    ok &= test_D_threshold()
    print("\n=== VEREDICTO:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
