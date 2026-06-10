"""Nivel 3 de validacion fisica: SUB-BLOQUES DE FUENTES con solucion analitica.

Aisla cada pieza del operador de fuentes rigidas (``implicit_source_solve``) y
del solve electrostatico (``solve_phi``) frente a soluciones cerradas, sobre
estados uniformes (sin transporte, sin contornos). Tres bloques:

A) GIRO MAGNETICO. Solo B (sin colisiones, sin E0, sin gravedad). La corriente
   ionica satisface, con q_i = +e, B = B \hat{y}:
       dj/dt = (e/M_i) (j x B),
   que en el plano (x, z) es rotacion a la girofrecuencia w_c = e B / M_i:
       j_x(t) =  j0 cos(w_c t),   j_z(t) = j0 sin(w_c t).
   El paso implicito (backward Euler) gira con direccion correcta y disipa
   ligeramente |j_perp| (esquema A-estable). Validamos angulo -> w_c t,
   direccion de giro y |j_perp| acotado.

B) RELAJACION COLISIONAL. Solo nu_in (sin B, sin E0, sin gravedad). El ion
   satisface dj/dt = -nu_in (j - e n_i U):
   - Con U = 0:  decaimiento exponencial j(t) = j0 exp(-nu_in t). El backward
     Euler da EXACTAMENTE j_N = j0 / (1 + nu_in dt)^N (test de precision de
     maquina del solver implicito), y converge a exp(-nu t) al refinar dt.
   - Con U != 0: equilibrio de deriva j* = e n_i U.

C) SOLVE DE POISSON. ``solve_phi`` resuelve ∇²φ = ∇·J (periodico). Para un modo
   J_x = sin(k x), ∇·J = k cos(k x) y la solucion exacta es φ = -(1/k) cos(k x).
   Validamos (1) φ frente a la solucion analitica (error de discretizacion) y
   (2) limpieza de divergencia: ∇·(J - ∇φ) ~ 0.

Ejecucion (WSL): MPIR_CVAR_ENABLE_GPU=0 python sources_epb.py
"""

import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from PyExner.solvers.kernels.epb_twofluid import EPBPhysParams, stack_state
from PyExner.solvers.kernels.epb_sources import (
    EPBSourceParams, implicit_source_solve, solve_phi,
    _central_diff, _laplacian, _div_backward, _grad_forward,
)
from PyExner.state.epb_twofluid_state import EPBTwoFluidState


PHYS = EPBPhysParams()   # e = M_i = M_e = 1


def _uniform_state(ni, ne, ji, je, shape=(2, 2)):
    """Estado uniforme con corrientes (ji, je) de 3 componentes cada una."""
    f = lambda v: jnp.full(shape, float(v))
    return EPBTwoFluidState(
        n_i=f(ni), n_e=f(ne),
        j_ix=f(ji[0]), j_iy=f(ji[1]), j_iz=f(ji[2]),
        j_ex=f(je[0]), j_ey=f(je[1]), j_ez=f(je[2]),
    )


def _ion_current(state):
    """Devuelve (jix, jiz) del primer punto (estado uniforme)."""
    return float(state.j_ix[0, 0]), float(state.j_iz[0, 0])


# --------------------------------------------------------------------------- #
# A) Giro magnetico                                                            #
# --------------------------------------------------------------------------- #

def test_magnetic_gyration():
    B = 1.0
    omega = PHYS.e * B / PHYS.Mi      # = 1
    j0 = 1.0
    t_final = 1.0                     # w_c t = 1 rad

    src = EPBSourceParams(By=B)       # solo magnetico; colisiones/E/g nulos

    # Direccion de giro y angulo, con malla fina en dt.
    print("[A magnetic] convergencia del angulo a w_c t = 1.000 rad:")
    errs = []
    resolutions = [50, 100, 200, 400]
    for nsteps in resolutions:
        dt = t_final / nsteps
        st = _uniform_state(1.0, 1.0, (j0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(nsteps):
            st = implicit_source_solve(st, dt, PHYS, src, dx=1.0)
        jx, jz = _ion_current(st)
        theta = math.atan2(jz, jx)
        err = abs(theta - omega * t_final)
        errs.append(err)
        print(f"  N={nsteps:4d}  jx={jx:+.5f} jz={jz:+.5f}  theta={theta:.5f}  err={err:.3e}")

    ok = True
    # Direccion de giro correcta: x -> z (jz > 0) tras 1 rad (~57 deg).
    if jz <= 0:
        print("[FAIL] direccion de giro incorrecta (jz<=0)"); ok = False
    # El angulo converge (error monotono decreciente y pequeno).
    if not all(errs[k] < errs[k - 1] for k in range(1, len(errs))):
        print("[FAIL] el error de angulo no decrece al refinar dt"); ok = False
    if errs[-1] > 5e-3:
        print(f"[FAIL] error de angulo final {errs[-1]:.2e} demasiado grande"); ok = False
    # |j_perp| disipado pero acotado (backward Euler A-estable: |j| <= j0).
    jperp = math.hypot(jx, jz)
    print(f"[A magnetic] |j_perp| final = {jperp:.5f} (j0 = {j0})")
    if not (0.9 * j0 <= jperp <= j0 + 1e-12):
        print(f"[FAIL] |j_perp| fuera de rango esperado: {jperp:.5f}"); ok = False
    print("[PASS] giro magnetico: direccion, angulo y |j_perp| correctos" if ok else "")
    return ok


# --------------------------------------------------------------------------- #
# B) Relajacion colisional                                                     #
# --------------------------------------------------------------------------- #

def test_collisional_decay():
    nu = 2.0
    j0 = 1.0
    t_final = 1.0
    src = EPBSourceParams(nu_in=nu)   # solo colision ion-neutro, U = 0

    ok = True

    # (B1) Backward Euler EXACTO: j_N = j0 / (1 + nu dt)^N (precision de maquina).
    nsteps = 100
    dt = t_final / nsteps
    st = _uniform_state(1.0, 1.0, (j0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for _ in range(nsteps):
        st = implicit_source_solve(st, dt, PHYS, src, dx=1.0)
    jx, _ = _ion_current(st)
    be_exact = j0 / (1.0 + nu * dt) ** nsteps
    print(f"[B decay] backward-Euler  num={jx:.10f}  formula={be_exact:.10f}")
    if abs(jx - be_exact) > 1e-9:
        print("[FAIL] el solver implicito no reproduce el backward Euler exacto"); ok = False

    # (B2) Convergencia a la solucion continua exp(-nu t) al refinar dt.
    exact_cont = j0 * math.exp(-nu * t_final)
    print(f"[B decay] convergencia a exp(-nu t) = {exact_cont:.6f}:")
    errs = []
    for nsteps in [50, 100, 200, 400]:
        dt = t_final / nsteps
        st = _uniform_state(1.0, 1.0, (j0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(nsteps):
            st = implicit_source_solve(st, dt, PHYS, src, dx=1.0)
        jx, _ = _ion_current(st)
        err = abs(jx - exact_cont)
        errs.append(err)
        print(f"  N={nsteps:4d}  jx={jx:.6f}  err={err:.3e}")
    if not all(errs[k] < errs[k - 1] for k in range(1, len(errs))):
        print("[FAIL] no converge a exp(-nu t) al refinar dt"); ok = False

    # (B3) Equilibrio de deriva: con U != 0, j -> e n_i U.
    Ux = 0.7
    src_drift = EPBSourceParams(nu_in=nu, Ux=Ux)
    dt = 0.05
    st = _uniform_state(1.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for _ in range(400):   # t = 20, nu t = 40 -> equilibrio
        st = implicit_source_solve(st, dt, PHYS, src_drift, dx=1.0)
    jx, _ = _ion_current(st)
    j_eq = PHYS.e * 1.0 * Ux
    print(f"[B drift] j_x -> {jx:.6f}  equilibrio e n U = {j_eq:.6f}")
    if abs(jx - j_eq) > 1e-4:
        print("[FAIL] no alcanza el equilibrio de deriva e n U"); ok = False

    if ok:
        print("[PASS] relajacion colisional: backward Euler exacto, convergencia y deriva")
    return ok


# --------------------------------------------------------------------------- #
# C) Solve de Poisson                                                          #
# --------------------------------------------------------------------------- #

def test_poisson_solve():
    L = 1.0
    nx = 128
    ny = 4
    m = 2                       # numero de modo
    k = 2.0 * math.pi * m / L
    dx = L / nx

    xc = (np.arange(nx) + 0.5) * dx
    Jx_1d = np.sin(k * xc)
    Jx = np.tile(Jx_1d, (ny, 1))           # (ny, nx), variacion solo en x
    zero = np.zeros((ny, nx))

    # Empaquetar como corriente ionica plana: arr[...,2]=Jx, arr[...,4]=Jz=0.
    # (solve_phi usa J = j_i + j_e en indices 2/5 (x) y 4/7 (z))
    arr = np.zeros((ny, nx, 8))
    arr[..., 0] = 1.0          # n_i
    arr[..., 1] = 1.0          # n_e
    arr[..., 2] = Jx           # j_ix
    arr = jnp.asarray(arr)

    phi = solve_phi(arr, dx, 4000)
    phi = np.asarray(phi)

    ok = True

    # (C1) φ frente a la solucion analitica continua φ = -(1/k) cos(k x).
    #   El solve de produccion (Paso 4) usa la tripleta compacta consistente
    #   (div backward / grad forward / L de 5 puntos): limpieza de divergencia
    #   EXACTA a cambio de exactitud O(dx) en el campo puntual (medio celda de
    #   desfase del RHS backward). Para m=2, nx=128 eso es ~5%.
    phi_exact = -(1.0 / k) * np.cos(k * xc)
    phi_exact -= phi_exact.mean()          # mismo gauge (media cero)
    phi_num = phi[0] - phi[0].mean()
    rel = np.max(np.abs(phi_num - phi_exact)) / np.max(np.abs(phi_exact))
    print(f"[C poisson] error relativo max |φ_num - φ_exact| = {rel:.3e}")
    if rel > 8e-2:
        print("[FAIL] φ no coincide con la solucion analitica"); ok = False

    # (C2) Residual del operador discreto CONSISTENTE: ∇²φ - ∇·J ~ 0 (~maquina).
    Jx_field = arr[..., 2] + arr[..., 5]
    Jz_field = arr[..., 4] + arr[..., 7]
    rhs = _div_backward(Jx_field, dx, 1) + _div_backward(Jz_field, dx, 0)
    lap = _laplacian(jnp.asarray(phi), dx)
    residual = float(jnp.max(jnp.abs(lap - rhs)))
    print(f"[C poisson] residual max |∇²φ - ∇·J| = {residual:.3e}")
    if residual > 1e-6:
        print("[FAIL] residual de Poisson demasiado grande"); ok = False

    # (C3) Limpieza de divergencia con el operador CONSISTENTE del solver.
    #   solve_phi resuelve ∇²_5pt φ = ∇·J (∇· = central_diff). La divergencia del
    #   campo corregido es ∇·J - ∇²_5pt φ (= residual C2). La reducimos varios
    #   ordenes frente a la divergencia original ||∇·J||.
    #   NOTA (riesgo numerico): si en su lugar se mide ∇·(J - ∇φ) componiendo dos
    #   derivadas centrales (∇· y ∇ de paso 2), aparece un modo "checkerboard"
    #   residual O(1e-2) porque div∘grad(central) != laplaciano de 5 puntos. Es
    #   el clasico desacoplamiento par/impar; para produccion conviene un
    #   gradiente/divergencia consistentes con el laplaciano (o malla escalonada).
    # (C3) Limpieza de divergencia con la tripleta CONSISTENTE del solver.
    #   El campo corregido es J - G^+φ (G^+ = _grad_forward, consistente con la
    #   divergencia backward del solve). Su divergencia D^-(J - G^+φ) = ∇·J - ∇²φ
    #   se anula a precision de maquina (Paso 4).
    div_orig = float(jnp.max(jnp.abs(rhs)))
    div_clean = float(jnp.max(jnp.abs(rhs - lap)))   # residual consistente
    reduction = div_clean / max(div_orig, 1e-30)
    print(f"[C poisson] ||∇·J||={div_orig:.3e}  ||∇·J - ∇²φ||={div_clean:.3e}  "
          f"reduccion={reduction:.3e}")
    if reduction > 1e-6:
        print("[FAIL] la proyeccion no reduce la divergencia (operador consistente)"); ok = False

    # (C3b) Verificacion directa del campo corregido J - G^+φ con la divergencia
    #   backward (lo que de verdad usa el momento): debe ser ~maquina. Con el
    #   gradiente CENTRAL viejo aparecia un modo checkerboard O(1e-2); Paso 4 lo
    #   resuelve usando grad forward consistente.
    Jx_star = Jx_field - _grad_forward(jnp.asarray(phi), dx, 1)
    Jz_star = Jz_field - _grad_forward(jnp.asarray(phi), dx, 0)
    div_consistent = float(jnp.max(jnp.abs(
        _div_backward(Jx_star, dx, 1) + _div_backward(Jz_star, dx, 0))))
    print(f"[C poisson] ||D^-(J - G^+φ)|| (campo corregido) = {div_consistent:.3e} "
          f"(checkerboard del Nivel 3 RESUELTO)")
    if div_consistent > 1e-6:
        print("[FAIL] el campo corregido no es divergente-libre consistente"); ok = False

    if ok:
        print("[PASS] solve de Poisson: φ analitico, residual y limpieza de divergencia")
    return ok


if __name__ == "__main__":
    print("=== Nivel 3: sub-bloques de fuentes (solucion analitica) ===\n")
    results = []
    results.append(test_magnetic_gyration()); print()
    results.append(test_collisional_decay()); print()
    results.append(test_poisson_solve()); print()
    npass = sum(results)
    print(f"{npass}/{len(results)} bloques OK")
    raise SystemExit(0 if npass == len(results) else 1)
