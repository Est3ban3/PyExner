"""Validacion del cierre ionosferico SI (camino critico, pasos 1-2).

Comprueba el modulo ``epb_ionosphere``: unidades fisicas SI, perfiles de fondo
de la region F ecuatorial, operador quimico produccion/recombinacion y el
diagnostico de tasa de crecimiento RT COLISIONAL. El objetivo es obtener
NUMEROS CONCRETOS comparables con observaciones de EPB (tiempos de e-folding de
~minutos, umbral de altura de aparicion).

Bloques:
  A) Unidades / parametros de fondo SI (sanidad fisica de valores).
  B) Equilibrio fotoquimico: P = beta n0 => la quimica deja n0 invariante.
  C) Operador quimico: relajacion exp(-beta t), positividad y estabilidad rigida.
  D) Resultado fisico CONCRETO: umbral de altura de la EPB via gamma colisional.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python ionosphere_epb.py
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_ionosphere import (
    SI_CONST,
    M_OPLUS,
    FRegionParams,
    build_background,
    physparams_SI,
    chemistry_implicit_step,
    density_scale_length,
    collisional_growth_rate,
)


# Columna vertical de prueba: 200-500 km, resolucion 2 km.
Z0 = 200.0e3
Z1 = 500.0e3
NZ = 151
Z = jnp.linspace(Z0, Z1, NZ)
DZ = float((Z1 - Z0) / (NZ - 1))

FR = FRegionParams()


def _idx_at(z_target):
    return int(np.argmin(np.abs(np.asarray(Z) - z_target)))


# --------------------------------------------------------------------------- #
# A) Unidades y parametros de fondo SI                                         #
# --------------------------------------------------------------------------- #

def test_units():
    print("--- A) Unidades / parametros de fondo SI ---")
    phys = physparams_SI(FR)
    ci = phys.ci
    ce = phys.ce
    bg = build_background(Z, FR)

    ip = _idx_at(FR.h_peak)
    i300 = _idx_at(300.0e3)

    print(f"  M(O+)        = {M_OPLUS:.3e} kg  (= {M_OPLUS / SI_CONST.amu:.1f} amu)")
    print(f"  c_i          = {ci:8.1f} m/s   (esperado ~0.7 km/s)")
    print(f"  c_e          = {ce:8.1f} m/s   (esperado ~120 km/s)")
    print(f"  n0(h_peak)   = {float(bg.n0[ip]):.3e} m^-3  (pico = {FR.n_max:.1e})")
    print(f"  nu_in(300km) = {float(bg.nu_in[i300]):.3e} s^-1 (esperado ~0.5)")
    print(f"  beta(300km)  = {float(bg.beta[i300]):.3e} s^-1")
    print(f"  B(350km)     = {float(bg.B[_idx_at(350e3)]) * 1e9:.0f} nT   (esperado ~2-3 e4 nT)")
    print(f"  g(350km)     = {float(bg.g[_idx_at(350e3)]):.3f} m/s^2 (esperado ~8.8)")

    ok = True
    # Velocidades del sonido en rango fisico.
    if not (500.0 < ci < 1500.0):
        print("  [FAIL] c_i fuera de rango fisico"); ok = False
    if not (5.0e4 < ce < 3.0e5):
        print("  [FAIL] c_e fuera de rango fisico"); ok = False
    # Pico de densidad en h_peak.
    if abs(float(bg.n0[ip]) - FR.n_max) / FR.n_max > 1e-6:
        print("  [FAIL] el pico de Chapman no esta en h_peak"); ok = False
    # B dipolar en rango de region F (1e4 - 4e4 nT).
    Bnt = float(bg.B[_idx_at(350e3)]) * 1e9
    if not (1.0e4 < Bnt < 4.0e4):
        print("  [FAIL] B fuera de rango ecuatorial"); ok = False
    # nu_in del orden de 1 s^-1 a 300 km.
    if not (0.3 < float(bg.nu_in[i300]) < 3.0):
        print("  [FAIL] nu_in(300km) fuera de rango"); ok = False
    print("  [OK] valores fisicos en rango" if ok else "  [FAIL] sanidad de unidades")
    return ok


# --------------------------------------------------------------------------- #
# B) Equilibrio fotoquimico: la quimica deja n0 invariante                     #
# --------------------------------------------------------------------------- #

def test_equilibrium():
    print("\n--- B) Equilibrio fotoquimico (P = beta n0) ---")
    bg = build_background(Z, FR)
    n_i = bg.n0
    n_e = bg.n0

    dt = 1.0  # s
    max_rel = 0.0
    for _ in range(50):
        n_i, n_e = chemistry_implicit_step(n_i, n_e, dt, bg.prod, bg.beta)
        rel = float(jnp.max(jnp.abs(n_i - bg.n0) / jnp.maximum(bg.n0, 1.0)))
        max_rel = max(max_rel, rel)

    print(f"  desviacion relativa maxima de n0 tras 50 pasos: {max_rel:.2e}")
    ok = max_rel < 1e-12
    print("  [OK] n0 es punto fijo exacto de la quimica" if ok
          else "  [FAIL] n0 no se conserva")
    return ok


# --------------------------------------------------------------------------- #
# C) Operador quimico: relajacion, positividad, rigidez                        #
# --------------------------------------------------------------------------- #

def test_chemistry_operator():
    print("\n--- C) Operador quimico (backward Euler positivo) ---")
    ok = True

    # C1) Relajacion exponencial hacia el equilibrio a tasa beta (celda unica).
    #     Sin produccion (P=0): n(t) = n0 exp(-beta t).
    beta = jnp.asarray(1.0e-3)        # s^-1
    n = jnp.asarray(1.0e12)
    prod = jnp.asarray(0.0)
    dt = 1.0
    N = 1000
    for _ in range(N):
        n, _ = chemistry_implicit_step(n, n, dt, prod, beta)
    analytic_BE = 1.0e12 / (1.0 + dt * float(beta)) ** N   # backward Euler exacto
    exact = 1.0e12 * np.exp(-float(beta) * dt * N)         # solucion continua
    err_BE = abs(float(n) - analytic_BE) / analytic_BE
    err_cont = abs(float(n) - exact) / exact
    print(f"  C1 relajacion: n={float(n):.4e}  BE-exacto={analytic_BE:.4e} "
          f"(err {err_BE:.1e})  continuo={exact:.4e} (err {err_cont:.1e})")
    if err_BE > 1e-10:
        print("  [FAIL] backward Euler no reproduce (1+dt beta)^-N"); ok = False
    if err_cont > 5e-2:
        print("  [FAIL] no converge a exp(-beta t)"); ok = False

    # C2) Estabilidad RIGIDA: beta enorme con dt grande sigue acotado y positivo.
    beta_stiff = jnp.asarray(1.0e6)   # recombinacion ultra-rapida
    n2 = jnp.asarray(1.0e12)
    prod2 = jnp.asarray(1.0e9)        # equilibrio n_eq = P/beta = 1e3
    for _ in range(100):
        n2, _ = chemistry_implicit_step(n2, n2, 1.0, prod2, beta_stiff)
    n_eq = float(prod2) / float(beta_stiff)
    print(f"  C2 rigidez beta=1e6 dt=1: n={float(n2):.4e}  n_eq=P/beta={n_eq:.4e} "
          f"(acotado, positivo)")
    if not (np.isfinite(float(n2)) and float(n2) > 0.0):
        print("  [FAIL] inestabilidad o densidad no positiva en regimen rigido"); ok = False
    if abs(float(n2) - n_eq) / n_eq > 1e-3:
        print("  [FAIL] no relaja al equilibrio P/beta"); ok = False

    # C3) Positividad sobre el perfil completo con perturbacion negativa fuerte.
    bg = build_background(Z, FR)
    n_i = bg.n0 * (1.0 - 0.9 * jnp.exp(-((Z - 250e3) ** 2) / (2 * (20e3) ** 2)))
    nmin = float(jnp.min(n_i))
    for _ in range(200):
        n_i, _ = chemistry_implicit_step(n_i, n_i, 1.0, bg.prod, bg.beta)
        nmin = min(nmin, float(jnp.min(n_i)))
    print(f"  C3 positividad perfil (deplecion 90%): min n = {nmin:.3e}")
    if nmin < 0.0:
        print("  [FAIL] densidad negativa"); ok = False

    print("  [OK] operador quimico A-estable y positivo" if ok
          else "  [FAIL] operador quimico")
    return ok


# --------------------------------------------------------------------------- #
# D) RESULTADO CONCRETO: umbral de altura de la EPB (gamma colisional)          #
# --------------------------------------------------------------------------- #

def test_growth_threshold():
    print("\n--- D) Umbral de altura de la EPB (gamma RT colisional) ---")
    bg = build_background(Z, FR)
    L_n = density_scale_length(bg.n0, DZ)
    gamma = collisional_growth_rate(bg.g, bg.nu_in, L_n, bg.beta)

    z_np = np.asarray(Z)
    g_np = np.asarray(gamma)
    Ln_np = np.asarray(L_n)
    nu_np = np.asarray(bg.nu_in)

    # Solo el bottomside (por debajo del pico) es fisicamente RT-relevante.
    bottom = z_np < FR.h_peak
    gpos = g_np > 0.0
    unstable = bottom & gpos

    print("  altura    L_n[km]   nu_in[s^-1]   gamma[s^-1]   tau=1/gamma")
    for h in (250e3, 275e3, 300e3, 325e3, 350e3):
        i = _idx_at(h)
        tau = 1.0 / g_np[i] if g_np[i] > 0 else np.inf
        tau_str = f"{tau / 60.0:6.1f} min" if np.isfinite(tau) else "  estable"
        print(f"  {h/1e3:5.0f} km  {Ln_np[i]/1e3:7.1f}  {nu_np[i]:10.3e}   "
              f"{g_np[i]:+10.3e}   {tau_str}")

    ok = True
    if unstable.any():
        z_thr = z_np[unstable].min()
        i_thr = np.where(z_np == z_thr)[0][0]
        # e-folding tipico en el bottomside inestable (mediana).
        tau_typ = np.median(1.0 / g_np[unstable]) / 60.0
        print(f"\n  UMBRAL de inestabilidad: h ~ {z_thr/1e3:.0f} km "
              f"(L_n={Ln_np[i_thr]/1e3:.1f} km)")
        print(f"  e-folding tipico en el bottomside inestable: tau ~ {tau_typ:.1f} min")
        # Comprobaciones fisicas concretas:
        # (1) existe un umbral por debajo del pico (no todo el dominio crece).
        if bottom.all() and unstable.all():
            print("  [WARN] todo el bottomside inestable (sin umbral claro)")
        # (2) el e-folding tipico esta en el rango observado (minutos a decenas).
        if not (0.5 < tau_typ < 120.0):
            print(f"  [FAIL] e-folding fuera del rango observado de EPB"); ok = False
        else:
            print(f"  [OK] e-folding en el rango observado de EPB (minutos)")
        # (3) la inestabilidad aparece preferentemente arriba (nu_in pequena).
        z_unstable_mean = z_np[unstable].mean()
        if z_unstable_mean < z_np[bottom].mean():
            print("  [FAIL] la inestabilidad no se concentra en la parte alta"); ok = False
        else:
            print("  [OK] inestabilidad concentrada donde nu_in es menor (parte alta)")
    else:
        print("  [FAIL] no hay region inestable (revisar parametros)"); ok = False

    return ok


def main():
    print("=== Validacion del cierre ionosferico SI (camino critico 1-2) ===")
    print(f"columna {Z0/1e3:.0f}-{Z1/1e3:.0f} km, dz={DZ/1e3:.1f} km, {NZ} niveles\n")
    results = [
        test_units(),
        test_equilibrium(),
        test_chemistry_operator(),
        test_growth_threshold(),
    ]
    ok = all(results)
    print("\n" + ("IONOSFERA SI: TODAS LAS PRUEBAS PASARON"
                  if ok else "IONOSFERA SI: HAY FALLOS"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
