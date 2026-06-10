"""Paso 7 del camino critico: disparador PRE y prediccion del onset de la EPB.

Valida ``epb_pre``: el ciclo diurno del drift vertical con el realce
pre-inversion (PRE), el decaimiento de la capa E al atardecer, la subida de la
capa F y el crecimiento RT acumulado. El resultado CONCRETO y comparable con
observaciones es la HORA LOCAL DEL ONSET de la burbuja.

Bloques (todos contra rangos observacionales, no contra numeros ajustados):

  A) DRIFT: pico del PRE en 18-19:30 LT con 20-60 m/s (Fejer et al.); drift
     diurno hacia arriba, nocturno hacia abajo.
  B) CAPA F: el PRE sube el pico F2 hasta 380-520 km entre 18:30 y 21 LT
     (radares/ionosondas: hmF2 ~ 400-500 km en tardes con PRE fuerte).
  C) ONSET: la tarde no acumula crecimiento (gamma <= 0 hasta las 17 LT y
     < 0.5 e-folds antes del terminador); gamma > 0 tras la puesta de sol; la
     amplificacion acumulada cruza ln(10^3) ~ 6.9 e-folds en 19-22 LT
     (ocurrencia observada de EPB: 19:30-22 LT).
  D) CONTROL SIN PRE: con V_pre = 0 (tarde sin realce) el onset se retrasa
     fuertemente o no ocurre — consistente con la correlacion observada
     PRE-EPB (el PRE es EL precursor estadistico dominante).

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python pre_onset_epb.py
"""

import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from PyExner.solvers.kernels.epb_pre import (
    GROWTH_THRESHOLD, PREParams, onset_prediction, vertical_drift,
)

P = PREParams()


def test_A_drift():
    t = np.linspace(0.0, 24.0, 2401)
    V = vertical_drift(t, P)
    k = int(np.argmax(V))
    t_pk, V_pk = float(t[k]), float(V[k])
    V_noon = float(vertical_drift(12.0, P))
    V_mid = float(vertical_drift(0.0, P))
    ok = (18.0 <= t_pk <= 19.5) and (20.0 <= V_pk <= 60.0) \
        and V_noon > 0.0 and V_mid < 0.0
    print(f"[A] PRE: pico {V_pk:.1f} m/s @ {t_pk:.2f} LT  "
          f"(obs: 20-60 m/s @ 18-19:30); mediodia {V_noon:+.1f}, "
          f"medianoche {V_mid:+.1f} m/s  {'PASS' if ok else 'FAIL'}")
    return ok


def test_C_onset(cast):
    ok = True
    t, h = cast.t_lt, cast.h_peak

    # B) subida de la capa
    k = int(np.argmax(h))
    h_max, t_hmax = float(h[k]) / 1e3, float(t[k])
    if not (380.0 <= h_max <= 520.0 and 18.5 <= t_hmax <= 21.0):
        print(f"[B][FAIL] capa: h_max = {h_max:.0f} km @ {t_hmax:.2f} LT"); ok = False
    else:
        print(f"[B][PASS] subida PRE de la capa F: h_max = {h_max:.0f} km @ "
              f"{t_hmax:.2f} LT (obs: 400-500 km tras el PRE)")

    # serie temporal resumida
    print("\n[C]   LT      V[m/s]   h_peak[km]   n_E[m^-3]   gamma[s^-1]   Gamma")
    for tq in (16.0, 17.0, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 24.0):
        k = int(np.argmin(np.abs(t - tq)))
        print(f"    {t[k]:5.2f}   {cast.V[k]:+7.1f}   {h[k] / 1e3:8.0f}   "
              f"{cast.n_E[k]:.2e}   {cast.gamma[k]:+.3e}   {cast.Gamma[k]:6.2f}")

    # tarde sin crecimiento apreciable: gamma <= 0 hasta las 17 LT y
    # acumulado < 0.5 e-folds antes del terminador (un gamma marginal
    # ~1e-5 s^-1, tau ~ dias, es fisicamente irrelevante).
    k17 = int(np.argmin(np.abs(t - 17.0)))
    k18 = int(np.argmin(np.abs(t - 18.0)))
    if not (np.all(cast.gamma[: k17 + 1] <= 0.0) and cast.Gamma[k18] < 0.5):
        print(f"[C][FAIL] crecimiento vespertino: Gamma(18 LT) = "
              f"{cast.Gamma[k18]:.2f} e-folds"); ok = False
    else:
        print(f"[C][PASS] tarde sin crecimiento: Gamma(18 LT) = "
              f"{cast.Gamma[k18]:.2f} e-folds (capa baja + capa E diurna)")

    # onset en la ventana observada
    if cast.t_onset is None or not (19.0 <= cast.t_onset <= 22.0):
        print(f"[C][FAIL] onset = {cast.t_onset} fuera de [19, 22] LT"); ok = False
    else:
        print(f"[C][PASS] ONSET predicho: {cast.t_onset:.2f} LT con "
              f"{GROWTH_THRESHOLD:.1f} e-folds (obs: 19:30-22 LT)")
    return ok


def test_D_control_no_pre(cast_pre):
    p0 = P._replace(V_pre=0.0)
    cast0 = onset_prediction(p0)
    if cast0.t_onset is None:
        msg = "sin onset antes de 02 LT"
        ok = True
    else:
        ok = cast0.t_onset >= cast_pre.t_onset + 1.5
        msg = f"onset retrasado a {cast0.t_onset:.2f} LT"
    efolds = float(cast0.Gamma[-1])
    print(f"[D] control SIN PRE: {msg}; e-folds acumulados a 02 LT = "
          f"{efolds:.2f} (con PRE: {float(cast_pre.Gamma[-1]):.1f})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=== Paso 7: disparador PRE y onset de la EPB ===\n")
    ok = True
    ok &= test_A_drift()
    cast = onset_prediction(P)
    ok &= test_C_onset(cast)
    print()
    ok &= test_D_control_no_pre(cast)
    print("\n=== VEREDICTO:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
