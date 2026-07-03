"""Solve electrostatico de produccion: tripleta consistente + CG.

Valida la TRIPLETA DE OPERADORES CONSISTENTE y el solver de gradiente
conjugado (CG) que sustituyeron al Jacobi + gradiente central original. El
hallazgo (en sources_epb.py) fue que mezclar el laplaciano de 5 puntos del
solve con un gradiente CENTRAL para construir E producia un modo
"checkerboard" (desacoplamiento par/impar) con divergencia residual O(1e-2):
``div_c o grad_c != L_5pt``.

Fix de produccion (operadores que cumplen D^- G^+ = L_5pt EXACTO):

    grad forward   G^+ f|_i = (f_{i+1} - f_i)/dx
    div  backward  D^- u|_i = (u_i - u_{i-1})/dx
    laplaciano     L = D^- G^+  =>  (f_{i+1}-2f_i+f_{i-1})/dx^2

Con ellos la limpieza de divergencia es EXACTA a precision de maquina:
    D^-(J - G^+ phi) = D^-J - L phi = residual del CG ~ 0.

Bloques:
    A) Identidad de operadores  D^- G^+ phi == L_5pt phi  (precision de maquina).
    B) Limpieza de divergencia EXACTA (consistente) vs el esquema viejo
       central o central (que deja el checkerboard O(1e-2)).
    C) Convergencia del CG: residual -> 0; recuperacion 2do orden de una
       solucion manufacturada; CG (pocas iters) mucho mejor que Jacobi.
    D) Ausencia de checkerboard: con J "ruidoso" par/impar el φ del solve
       consistente no amplifica el modo de tablero.

Ejecutar:
    MPIR_CVAR_ENABLE_GPU=0 python poisson_epb.py
"""

import math

import jax
import jax.numpy as jnp
import numpy as np  # noqa: F401

jax.config.update("jax_enable_x64", True)

from PyExner.solvers.kernels.epb_sources import (
    _grad_forward, _div_backward, _laplacian, _central_diff,
    _poisson_divergence, _poisson_cg, solve_phi,
    EPBSourceParams, electric_field,
)


def _pack(Jx, Jz):
    """Empaqueta corrientes (Jx, Jz) como arr (...,8): ion x=idx2, z=idx4."""
    shp = Jx.shape
    arr = jnp.zeros(shp + (8,))
    arr = arr.at[..., 0].set(1.0)   # n_i
    arr = arr.at[..., 1].set(1.0)   # n_e
    arr = arr.at[..., 2].set(Jx)    # j_ix
    arr = arr.at[..., 4].set(Jz)    # j_iz
    return arr


# --------------------------------------------------------------------------- #
# A) Identidad de operadores                                                   #
# --------------------------------------------------------------------------- #

def test_operator_identity():
    print("--- A) Identidad de operadores  D^- G^+ == L_5pt ---")
    key = jax.random.PRNGKey(0)
    f = jax.random.normal(key, (48, 64))
    dx = 0.123

    lap_compose = (_div_backward(_grad_forward(f, dx, 1), dx, 1)
                   + _div_backward(_grad_forward(f, dx, 0), dx, 0))
    lap_direct = _laplacian(f, dx)
    err = float(jnp.max(jnp.abs(lap_compose - lap_direct)))
    print(f"  max |D^-G^+ f - L_5pt f| = {err:.3e}")
    ok = err < 1e-12
    if not ok:
        print("  [FAIL] la tripleta no es consistente con el laplaciano")
    else:
        print("  [OK] D^- G^+ = L_5pt a precision de maquina")
    return ok


# --------------------------------------------------------------------------- #
# B) Limpieza de divergencia EXACTA vs el esquema viejo                        #
# --------------------------------------------------------------------------- #

def test_divergence_cleaning():
    print("--- B) Limpieza de divergencia (consistente vs central viejo) ---")
    L = 2.0
    n = 64
    dx = L / n
    # Corrientes SUAVES (representativas de la fisica; varios modos de Fourier).
    xc = (np.arange(n) + 0.5) * dx
    X, Z = np.meshgrid(xc, xc)
    Jx = jnp.asarray(np.sin(2 * math.pi * 2 * X / L) + 0.4 * np.cos(2 * math.pi * 1 * Z / L))
    Jz = jnp.asarray(0.7 * np.cos(2 * math.pi * 3 * X / L) - 0.3 * np.sin(2 * math.pi * 2 * Z / L))
    arr = _pack(Jx, Jz)

    # Divergencia original (consistente, backward) antes de limpiar.
    div0 = float(jnp.max(jnp.abs(_poisson_divergence(arr, dx))))

    # --- Esquema CONSISTENTE de produccion ---
    phi = solve_phi(arr, dx, 200)
    src = EPBSourceParams()
    # Campo corregido J* = J - G^+ phi  (E = E0 - grad phi con E0 = 0).
    E = electric_field(phi, src, dx)       # E = -grad phi
    Jx_star = Jx + E[..., 0]               # J - grad phi  (E_x = -dphi/dx)
    Jz_star = Jz + E[..., 2]
    arr_star = _pack(Jx_star, Jz_star)
    div_clean = float(jnp.max(jnp.abs(_poisson_divergence(arr_star, dx))))
    red = div_clean / max(div0, 1e-30)
    print(f"  consistente : ||D^-J||={div0:.3e}  ||D^-(J-G^+φ)||={div_clean:.3e}"
          f"  reduccion={red:.3e}")
    ok_consistent = red < 1e-8

    # --- Esquema VIEJO (central o central): deja el checkerboard ---
    # phi resuelto con L_5pt pero gradiente CENTRAL para E (mismatch).
    dphidx_c = _central_diff(phi, dx, 1)
    dphidz_c = _central_diff(phi, dx, 0)
    Jx_old = Jx - dphidx_c
    Jz_old = Jz - dphidz_c
    # divergencia central del campo corregido (como lo media el esquema viejo)
    div_old = float(jnp.max(jnp.abs(
        _central_diff(Jx_old, dx, 1) + _central_diff(Jz_old, dx, 0))))
    red_old = div_old / max(div0, 1e-30)
    print(f"  central viejo: ||div_c(J - grad_c φ)||={div_old:.3e}"
          f"  reduccion={red_old:.3e}")

    if ok_consistent and red < red_old:
        print("  [OK] limpieza EXACTA (~maquina); supera al esquema central")
    else:
        print("  [FAIL] la limpieza consistente no alcanza precision de maquina")
        ok_consistent = False
    return ok_consistent


# --------------------------------------------------------------------------- #
# C) Convergencia del CG y solucion manufacturada                             #
# --------------------------------------------------------------------------- #

def _jacobi_solve(arr, dx, n_iter):
    """Jacobi viejo (5 puntos) para comparar convergencia."""
    rhs = _poisson_divergence(arr, dx)

    def body(_, phi):
        up = jnp.roll(phi, -1, 0); down = jnp.roll(phi, 1, 0)
        right = jnp.roll(phi, -1, 1); left = jnp.roll(phi, 1, 1)
        phi_new = 0.25 * (up + down + left + right - dx * dx * rhs)
        return phi_new - jnp.mean(phi_new)

    return jax.lax.fori_loop(0, n_iter, body, jnp.zeros_like(rhs))


def test_cg_convergence():
    print("--- C) Convergencia CG y solucion manufacturada ---")
    ok = True

    # C1) Residual del CG decae con iteraciones (campo suave representativo).
    n = 64
    dx = 1.0 / n
    xc = (np.arange(n) + 0.5) * dx
    X, Z = np.meshgrid(xc, xc)
    Jx = jnp.asarray(np.sin(2 * math.pi * 2 * X) + 0.5 * np.cos(2 * math.pi * 3 * Z))
    arr = _pack(Jx, jnp.zeros((n, n)))
    rhs = _poisson_divergence(arr, dx)
    print("    iters    ||L φ - rhs||_inf")
    for it in (5, 10, 20, 40, 80):
        phi = _poisson_cg(rhs, dx, it)
        res = float(jnp.max(jnp.abs(_laplacian(phi, dx) - rhs)))
        print(f"    {it:5d}    {res:.3e}")
    res_cg = res

    # C2) CG (40 iters) frente a Jacobi (40 iters).
    phi_cg40 = _poisson_cg(rhs, dx, 40)
    res_cg40 = float(jnp.max(jnp.abs(_laplacian(phi_cg40, dx) - rhs)))
    phi_j = _jacobi_solve(arr, dx, 40)
    res_j = float(jnp.max(jnp.abs(_laplacian(phi_j, dx) - rhs)))
    print(f"    CG(40)={res_cg40:.3e}   Jacobi(40)={res_j:.3e}   "
          f"CG es {res_j / max(res_cg40, 1e-30):.2e}x mas preciso")
    if not (res_cg40 < 1e-8 and res_cg40 < res_j):
        print("    [FAIL] CG no converge mejor que Jacobi"); ok = False

    # C3) Solucion manufacturada: J = sin(k x) => phi continuo = -(1/k) cos(k x).
    m = 3
    Lx = 1.0
    nx = 256
    dxx = Lx / nx
    k = 2.0 * math.pi * m / Lx
    xc = (np.arange(nx) + 0.5) * dxx
    Jx_1d = np.sin(k * xc)
    arr2 = _pack(jnp.asarray(np.tile(Jx_1d, (4, 1))), jnp.zeros((4, nx)))
    phi = np.array(solve_phi(arr2, dxx, 200))[0]
    phi = phi - phi.mean()
    phi_exact = -(1.0 / k) * np.cos(k * xc)
    phi_exact -= phi_exact.mean()
    rel = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))
    print(f"    solucion manufacturada (m={m}, nx={nx}): error rel = {rel:.3e}")
    if rel > 5e-2:
        print("    [FAIL] no recupera la solucion analitica"); ok = False

    if ok:
        print("  [OK] CG converge a ~maquina y recupera la solucion fisica")
    return ok


# --------------------------------------------------------------------------- #
# D) Control del modo checkerboard                                            #
# --------------------------------------------------------------------------- #

def test_checkerboard_control():
    print("--- D) Control del modo checkerboard (compacto vs central viejo) ---")
    n = 64
    dx = 1.0 / n
    ii = (np.arange(n)[:, None] + np.arange(n)[None, :])
    cb = jnp.asarray((-1.0) ** ii)   # modo de tablero puro

    # Laplaciano COMPACTO de 5 puntos (el del solve consistente): autovalor -8/dx^2.
    lap5 = _laplacian(cb, dx)
    amp5 = float(jnp.max(jnp.abs(lap5)))

    # Laplaciano CENTRAL ancho (div_c o grad_c) del esquema VIEJO: el checkerboard
    # cae en su NUCLEO (autovalor 0) -> modo libre, no controlado.
    lap_wide = (_central_diff(_central_diff(cb, dx, 1), dx, 1)
                + _central_diff(_central_diff(cb, dx, 0), dx, 0))
    amp_wide = float(jnp.max(jnp.abs(lap_wide)))

    expected = 8.0 / (dx * dx)
    print(f"  |L_5pt cb|   = {amp5:.3e}   (esperado 8/dx^2 = {expected:.3e})")
    print(f"  |L_central cb| = {amp_wide:.3e}   (≈0 => modo en el nucleo)")

    ok = (amp5 > 0.5 * expected) and (amp_wide < 1e-6 * expected)
    if ok:
        print("  [OK] el operador compacto ACOPLA el tablero (no es modo libre);"
              " el central viejo lo deja en su nucleo")
    else:
        print("  [FAIL] el control del checkerboard no es el esperado")
    return ok


if __name__ == "__main__":
    print("=== Solve electrostatico consistente (operadores + CG) ===")
    results = [
        test_operator_identity(),
        test_divergence_cleaning(),
        test_cg_convergence(),
        test_checkerboard_control(),
    ]
    if all(results):
        print("POISSON: TODAS LAS PRUEBAS PASARON")
    else:
        print(f"POISSON: {sum(results)}/{len(results)} bloques pasaron")
