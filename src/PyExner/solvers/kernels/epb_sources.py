# PyExner/solvers/kernels/epb_sources.py
"""Fuentes rigidas S(Q) y solve electrostatico (rama EPB_TwoFluid).

Separado a proposito del kernel de transporte (epb_twofluid.py): el paso
hiperbolico no tiene por que saber nada de colisiones ni del solve de phi.

Las variables j_alpha son densidades de CORRIENTE (j_i = e n_i v_i,
j_e = -e n_e v_e). Multiplicando la ecuacion de momento de cada especie por
q_alpha/M_alpha (el mismo escalado que mete el +-(e/M) p en el flujo) queda:

    S_{j_i} =  e n_i g + (e^2 n_i / M_i) E + (e/M_i) j_i x B
               - nu_in (j_i - e n_i U)
    S_{j_e} = -e n_e g + (e^2 n_e / M_e) E - (e/M_e) j_e x B
               - nu_en (j_e + e n_e U) - nu_ei (j_e + (n_e/n_i) j_i)

con q_i = +e, q_e = -e. Las filas de continuidad no llevan fuente (la quimica
va aparte, en epb_ionosphere.py).

El acoplamiento electrostatico es tipo proyeccion: se pide div(J) = 0 con
J = j_i + j_e (componentes del plano x,z), se resuelve lap(phi) = div(J) y el
campo efectivo es E = E_0 - grad(phi).

Orden de un paso completo: transporte -> solve phi -> E -> fuentes implicitas.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from functools import partial

from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams,
    stack_state,
    unstack_state,
    DENS_FLOOR,
)


class EPBSourceParams(NamedTuple):
    """Campos de fondo y frecuencias de colision del operador de fuentes.

    Geometria 2.5D: los vectores tienen tres componentes pero solo x y z se
    discretizan. Todo en cero por defecto, asi el transporte puro sigue
    siendo el caso base y no se altera nada de lo ya validado.
    """
    # Gravedad g = (gx, gy, gz)
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    # Campo electrico de fondo E_0 = (E0x, E0y, E0z)
    E0x: float = 0.0
    E0y: float = 0.0
    E0z: float = 0.0
    # Campo magnetico B = (Bx, By, Bz). En geometria ecuatorial suele ir fuera
    # del plano (eje y); por defecto nulo.
    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0
    # Viento neutro de fondo U = (Ux, Uy, Uz)
    Ux: float = 0.0
    Uy: float = 0.0
    Uz: float = 0.0
    # Frecuencias de colision
    nu_in: float = 0.0   # ion - neutro
    nu_en: float = 0.0   # electron - neutro
    nu_ei: float = 0.0   # electron - ion
    # Iteraciones del solve de Poisson (parametro numerico, no fisico).
    poisson_iters: int = 200

    @property
    def g(self):
        return jnp.array([self.gx, self.gy, self.gz])

    @property
    def E0(self):
        return jnp.array([self.E0x, self.E0y, self.E0z])

    @property
    def B(self):
        return jnp.array([self.Bx, self.By, self.Bz])

    @property
    def U(self):
        return jnp.array([self.Ux, self.Uy, self.Uz])

    @classmethod
    def from_params(cls, params: dict) -> "EPBSourceParams":
        src = ((params or {}).get("epb", {}) or {}).get("sources", {}) or {}
        d = cls()
        return cls(
            gx=float(src.get("gx", d.gx)), gy=float(src.get("gy", d.gy)), gz=float(src.get("gz", d.gz)),
            E0x=float(src.get("E0x", d.E0x)), E0y=float(src.get("E0y", d.E0y)), E0z=float(src.get("E0z", d.E0z)),
            Bx=float(src.get("Bx", d.Bx)), By=float(src.get("By", d.By)), Bz=float(src.get("Bz", d.Bz)),
            Ux=float(src.get("Ux", d.Ux)), Uy=float(src.get("Uy", d.Uy)), Uz=float(src.get("Uz", d.Uz)),
            nu_in=float(src.get("nu_in", d.nu_in)),
            nu_en=float(src.get("nu_en", d.nu_en)),
            nu_ei=float(src.get("nu_ei", d.nu_ei)),
            poisson_iters=int(src.get("poisson_iters", d.poisson_iters)),
        )


# --------------------------------------------------------------------------- #
# Vector fuente S(Q)                                                           #
# --------------------------------------------------------------------------- #

def source_term(arr: jax.Array, phys: EPBPhysParams, src: EPBSourceParams, E: jax.Array) -> jax.Array:
    """Vector fuente S(Q) de forma (..., 8) en el orden canonico.

    Las filas de continuidad van en cero: las fuentes solo tocan corrientes.
    ``E`` es el campo efectivo por celda, (..., 3), tipicamente E_0 - grad(phi).
    """
    ni = arr[..., 0]
    ne = arr[..., 1]
    ji = arr[..., 2:5]   # (jix, jiy, jiz)
    je = arr[..., 5:8]   # (jex, jey, jez)

    e, Mi, Me = phys.e, phys.Mi, phys.Me
    g, B, U = src.g, src.B, src.U

    ni_ = ni[..., None]
    ne_ = ne[..., None]

    # --- Bloque ionico ---
    S_i = (
        (e * ni_) * g                       # gravedad
        + (e * e * ni_ / Mi) * E            # fuerza electrica
        + (e / Mi) * jnp.cross(ji, B)       # fuerza magnetica
        - src.nu_in * (ji - e * ni_ * U)    # colision ion-neutro
    )

    # --- Bloque electronico ---
    ratio = ne_ / jnp.maximum(ni_, DENS_FLOOR)
    S_e = (
        (-e * ne_) * g                      # gravedad
        + (e * e * ne_ / Me) * E            # fuerza electrica
        - (e / Me) * jnp.cross(je, B)       # fuerza magnetica
        - src.nu_en * (je + e * ne_ * U)    # colision electron-neutro
        - src.nu_ei * (je + ratio * ji)     # colision electron-ion
    )

    zeros = jnp.zeros_like(ni)
    return jnp.stack(
        [
            zeros, zeros,
            S_i[..., 0], S_i[..., 1], S_i[..., 2],
            S_e[..., 0], S_e[..., 1], S_e[..., 2],
        ],
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# Solve electrostatico: potencial de polarizacion y campo efectivo            #
# --------------------------------------------------------------------------- #

def _central_diff(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Derivada central de 2do orden, bordes periodicos.

    Ya no se usa en el solve electrostatico: la central no compone bien con el
    laplaciano de 5 puntos (div_c o grad_c != L_5) y eso generaba el modo
    checkerboard. La dejo solo como herramienta de diagnostico.
    """
    fwd = jnp.roll(field, -1, axis=axis)
    bwd = jnp.roll(field, 1, axis=axis)
    return (fwd - bwd) / (2.0 * dx)


# --------------------------------------------------------------------------- #
# Tripleta de operadores consistente: D^- G^+ = L_5 exacto.                    #
#                                                                             #
#   grad forward   G^+ f|_i = (f_{i+1} - f_i)/dx        (vive en la cara)      #
#   div  backward  D^- u|_i = (u_i - u_{i-1})/dx        (adjunta de -G^+)      #
#   laplaciano     L = D^- G^+ = (f_{i+1} - 2 f_i + f_{i-1})/dx^2              #
#                                                                             #
# Con esto la limpieza de divergencia es exacta a precision de maquina:       #
#   D^-(J - G^+ phi) = D^- J - L phi = residual del CG ~ 0.                    #
# Aprendido a la mala: mezclar laplaciano de 5 puntos con gradiente central   #
# desacopla pares e impares y phi sale con checkerboard.                      #
# --------------------------------------------------------------------------- #

def _grad_forward(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Gradiente forward G^+ f|_i = (f_{i+1} - f_i)/dx (periodico)."""
    return (jnp.roll(field, -1, axis=axis) - field) / dx


def _div_backward(field: jax.Array, dx: float, axis: int) -> jax.Array:
    """Divergencia backward D^- u|_i = (u_i - u_{i-1})/dx (adjunta de -G^+)."""
    return (field - jnp.roll(field, 1, axis=axis)) / dx


def _laplacian(field: jax.Array, dx: float) -> jax.Array:
    """Laplaciano de 5 puntos (axis 0 = z, axis 1 = x).

    Coincide EXACTAMENTE con D^-(G^+ f) en ambas direcciones; esa identidad
    es la que hace consistente todo el solve.
    """
    up = jnp.roll(field, -1, axis=0)
    down = jnp.roll(field, 1, axis=0)
    right = jnp.roll(field, -1, axis=1)
    left = jnp.roll(field, 1, axis=1)
    return (up + down + left + right - 4.0 * field) / (dx * dx)


def _poisson_divergence(arr: jax.Array, dx: float) -> jax.Array:
    """RHS del Poisson: div(J) con la divergencia backward.

    J = j_i + j_e en el plano (x: indices 2 y 5; z: 4 y 7). Se le quita la
    media porque el sistema periodico solo es compatible con RHS de media cero.
    """
    Jx = arr[..., 2] + arr[..., 5]   # jix + jex
    Jz = arr[..., 4] + arr[..., 7]   # jiz + jez
    rhs = _div_backward(Jx, dx, axis=1) + _div_backward(Jz, dx, axis=0)
    return rhs - jnp.mean(rhs)


@partial(jax.jit, static_argnums=(2,))
def _poisson_cg(rhs: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Gradiente conjugado para -L phi = -rhs (L = laplaciano de 5 puntos).

    A = -L es simetrica y semidefinida positiva; el nucleo son las constantes
    (BC periodicas), asi que se trabaja en media cero (se proyecta A p en cada
    matvec) y al final se fija el gauge <phi> = 0. n_iter es estatico para que
    el fori_loop sea jiteable.

    Detalle importante: cuando el residual relativo baja de 1e-12 la iteracion
    se congela (alpha = beta = 0). Si se sigue iterando despues de converger
    se termina dividiendo por ~0 y la solucion se ensucia; con la congelacion
    puedo sobredimensionar n_iter sin miedo.
    """
    def Aop(p):
        ap = -_laplacian(p, dx)
        return ap - jnp.mean(ap)

    b = -(rhs - jnp.mean(rhs))
    phi = jnp.zeros_like(rhs)
    r = b - Aop(phi)          # = b (phi0 = 0)
    p = r
    rs = jnp.sum(r * r)
    rs0 = jnp.maximum(rs, 1e-300)

    def body(_, carry):
        phi, r, p, rs = carry
        Ap = Aop(p)
        denom = jnp.sum(p * Ap)
        # Solo iterar mientras no haya convergido y la curvatura sea util.
        active = (rs > 1e-24 * rs0) & (denom > 1e-300)
        alpha = jnp.where(active, rs / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        phi = phi + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        beta = jnp.where(active, rs_new / jnp.where(rs > 0.0, rs, 1.0), 0.0)
        p = jnp.where(active, r + beta * p, p)
        rs_out = jnp.where(active, rs_new, rs)
        return (phi, r, p, rs_out)

    phi, r, p, rs = jax.lax.fori_loop(0, n_iter, body, (phi, r, p, rs))
    return phi - jnp.mean(phi)


@partial(jax.jit, static_argnums=(2,))
def solve_phi(arr: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Potencial de polarizacion: resuelve lap(phi) = div(J), estilo proyeccion.

    Tripleta consistente (D^- / G^+ / L_5) + CG. Ejes: axis 1 = x, axis 0 = z.
    BC periodicas (todo con jnp.roll).

    La garantia que importa: la corriente corregida J - G^+ phi queda libre de
    divergencia (en el sentido D^-) a precision de maquina, porque
    D^-(J - G^+ phi) = rhs - L phi = residual del CG. Nada de checkerboard.

    Pendiente para produccion: contornos no periodicos y version MPI (haria
    falta halo_exchange dentro de cada matvec del CG); la tripleta y el CG no
    cambiarian.
    """
    rhs = _poisson_divergence(arr, dx)
    return _poisson_cg(rhs, dx, n_iter)


def electric_field(phi: jax.Array, src: EPBSourceParams, dx: float) -> jax.Array:
    """Campo efectivo E = E_0 - grad(phi), de forma (..., 3).

    El gradiente es el forward (G^+), el mismo del solve; usar otro gradiente
    aca rompe la limpieza exacta de la divergencia. Solo se corrigen E_x y
    E_z; E_y (fuera del plano) se queda en su valor de fondo.
    """
    dphidx = _grad_forward(phi, dx, axis=1)
    dphidz = _grad_forward(phi, dx, axis=0)

    Ex = src.E0x - dphidx
    Ey = jnp.full_like(phi, src.E0y)
    Ez = src.E0z - dphidz
    return jnp.stack([Ex, Ey, Ez], axis=-1)


# --------------------------------------------------------------------------- #
# Solve electrostatico con conductancia variable (carga de la capa E)          #
#                                                                             #
# Fisica: integrando en el tubo de flujo, la ecuacion del potencial es        #
#                                                                             #
#     div( (Sigma_F + Sigma_E) grad phi ) = div J_drive                       #
#                                                                             #
# con Sigma_F ~ n (Pedersen local de la region F) y Sigma_E la conductancia   #
# de las capas E conjugadas. La capa E es un cortocircuito en paralelo: drena #
# la carga de polarizacion a lo largo de B y diluye el campo por el factor    #
# F_s = Sigma_F/(Sigma_F + Sigma_E). Por eso la burbuja solo crece de noche.  #
#                                                                             #
# Numerica: la misma tripleta consistente pero con sigma evaluada en caras    #
# (media aritmetica):  L_sigma phi = D^-( sigma_{i+1/2} G^+ phi ).            #
# Como cada cara comparte su sigma con las dos celdas, A = -L_sigma sigue     #
# siendo simetrica y SPD (en media cero) para sigma > 0, y el mismo CG        #
# funciona sin tocar nada. Chequeo util: con sigma uniforme = c la solucion   #
# es exactamente phi_cte/c. sigma va normalizada a la conductancia F de       #
# referencia, asi que sigma = 1 reproduce el solve original.                  #
# --------------------------------------------------------------------------- #

SIGMA_FLOOR = 1e-12   # piso de sigma: mantiene el operador SPD si n -> 0 dentro de la burbuja


def _face_avg(sigma: jax.Array, axis: int) -> jax.Array:
    """sigma en la cara i+1/2: media aritmetica (sigma_i + sigma_{i+1})/2."""
    return 0.5 * (sigma + jnp.roll(sigma, -1, axis=axis))


def _div_sigma_grad(phi: jax.Array, sigma: jax.Array, dx: float) -> jax.Array:
    """Operador de coeficiente variable L_sigma phi = D^-(sigma_f G^+ phi).

    Con sigma = 1 reduce exactamente a ``_laplacian``. Simetrico (la sigma de
    cara es compartida) y definido negativo en media cero para sigma > 0.
    """
    fx = _face_avg(sigma, 1) * _grad_forward(phi, dx, axis=1)
    fz = _face_avg(sigma, 0) * _grad_forward(phi, dx, axis=0)
    return _div_backward(fx, dx, axis=1) + _div_backward(fz, dx, axis=0)


@partial(jax.jit, static_argnums=(3,))
def _poisson_cg_sigma(rhs: jax.Array, sigma: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """CG para -L_sigma phi = -rhs. Copia de ``_poisson_cg`` con el operador
    de coeficiente variable: mismo subespacio de media cero, mismo gauge y
    misma congelacion al converger. La mantengo separada para no tocar el
    kernel de coeficiente constante ya validado.
    """
    sig = jnp.maximum(sigma, SIGMA_FLOOR)

    def Aop(p):
        ap = -_div_sigma_grad(p, sig, dx)
        return ap - jnp.mean(ap)

    b = -(rhs - jnp.mean(rhs))
    phi = jnp.zeros_like(rhs)
    r = b - Aop(phi)
    p = r
    rs = jnp.sum(r * r)
    rs0 = jnp.maximum(rs, 1e-300)

    def body(_, carry):
        phi, r, p, rs = carry
        Ap = Aop(p)
        denom = jnp.sum(p * Ap)
        active = (rs > 1e-24 * rs0) & (denom > 1e-300)
        alpha = jnp.where(active, rs / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        phi = phi + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        beta = jnp.where(active, rs_new / jnp.where(rs > 0.0, rs, 1.0), 0.0)
        p = jnp.where(active, r + beta * p, p)
        rs_out = jnp.where(active, rs_new, rs)
        return (phi, r, p, rs_out)

    phi, r, p, rs = jax.lax.fori_loop(0, n_iter, body, (phi, r, p, rs))
    return phi - jnp.mean(phi)


@partial(jax.jit, static_argnums=(3,))
def solve_phi_sigma(arr: jax.Array, sigma: jax.Array, dx: float, n_iter: int) -> jax.Array:
    """Potencial con conductancia variable: D^-(sigma G^+ phi) = D^- J.

    ``sigma`` es adimensional (normalizada a la conductancia F de referencia);
    lo tipico es sigma = n/n_ref + R_E, con R_E = Sigma_E/Sigma_ref el shunt
    de la capa E. Hereda la garantia del solve constante: D^-J - L_sigma phi =
    residual del CG ~ 0. BC periodicas.
    """
    rhs = _poisson_divergence(arr, dx)
    return _poisson_cg_sigma(rhs, sigma, dx, n_iter)


# --------------------------------------------------------------------------- #
# Aplicacion explicita (solo para pruebas)                                     #
# --------------------------------------------------------------------------- #

def apply_sources_explicit(
    state, dt: float, phys: EPBPhysParams, src: EPBSourceParams, dx: float
):
    """Paso de fuentes Euler explicito Q <- Q + dt S(Q).

    Sirve para validar el operador de fuentes aislado, nada mas. En produccion
    no se usa: las colisiones y la girofrecuencia son rigidas y esto explota
    con el dt convectivo; el paso serio es ``implicit_source_solve`` via IMEX.
    Orden interno: solve phi -> E -> S(Q) -> update.
    """
    arr = stack_state(state)
    phi = solve_phi(arr, dx, src.poisson_iters)
    E = electric_field(phi, src, dx)
    S = source_term(arr, phys, src, E)
    return unstack_state(arr + dt * S)


# --------------------------------------------------------------------------- #
# Solve implicito de las fuentes rigidas (la parte "Im" del IMEX)              #
# --------------------------------------------------------------------------- #

def _cross_matrix(Bx: float, By: float, Bz: float) -> jax.Array:
    """Matriz K tal que K j = j x B."""
    return jnp.array(
        [
            [0.0, Bz, -By],
            [-Bz, 0.0, Bx],
            [By, -Bx, 0.0],
        ]
    )


def implicit_source_solve(
    state, dt: float, phys: EPBPhysParams, src: EPBSourceParams, dx: float,
    sigma: jax.Array | None = None, E_ext: jax.Array | None = None,
):
    """Backward Euler de las fuentes rigidas (parte implicita del IMEX).

    Aprovecho que S es afin en las corrientes (las densidades no se tocan):

        S(j) = A j + b

    con bloques (q_i = +e, q_e = -e, K j = j x B):

        A_ion = (e/M_i) K - nu_in I
        A_ee  = -(e/M_e) K - (nu_en + nu_ei) I
        A_ei  = -nu_ei (n_e/n_i) I        (el electron siente al ion)

        b_i =  e n_i g + (e^2 n_i/M_i) E + nu_in e n_i U
        b_e = -e n_e g + (e^2 n_e/M_e) E - nu_en e n_e U

    Entonces cada celda resuelve su sistema 6x6

        (I - dt A) j^{n+1} = j^n + dt b

    vectorizado con jnp.linalg.solve. A es bloque-triangular inferior (el ion
    no depende del electron), asi que el sistema esta bien condicionado para
    dt razonables. Sin esto no hay forma: Omega dt >> 1 mata cualquier esquema
    explicito, y el backward Euler es A-estable.

    El campo E entra congelado (lagged): se resuelve phi con el estado actual
    y ese E se usa en b. La rigidez local queda implicita; el acoplamiento
    electrostatico global queda explicito. Funciona porque phi evoluciona en
    la escala lenta del transporte, no en la de las colisiones.

    Argumentos opcionales:
      * sigma: conductancia adimensional (p.ej. n_i/n_ref + R_E). Si viene,
        phi se resuelve con solve_phi_sigma; si no, con solve_phi de siempre.
      * E_ext: campo TOTAL precalculado (..., 3). Si viene, aca no se resuelve
        phi: el caller se hace cargo del cierre electrostatico. Lo necesita el
        cierre drive-driven de plume_si_epb: alli el RHS correcto es solo la
        corriente motriz, y resolver con la J total es inestable porque el
        estado ya arrastra la respuesta Pedersen sigma_P E del paso anterior.
    """
    arr = stack_state(state)
    ni = arr[..., 0]
    ne = arr[..., 1]
    ji = arr[..., 2:5]
    je = arr[..., 5:8]

    # Campo electrico congelado en el paso (acoplamiento explicito).
    if E_ext is not None:
        E = E_ext
    elif sigma is None:
        phi = solve_phi(arr, dx, src.poisson_iters)
        E = electric_field(phi, src, dx)   # (..., 3)
    else:
        phi = solve_phi_sigma(arr, sigma, dx, src.poisson_iters)
        E = electric_field(phi, src, dx)   # (..., 3)

    e, Mi, Me = phys.e, phys.Mi, phys.Me
    K = _cross_matrix(src.Bx, src.By, src.Bz)   # (3, 3)
    I3 = jnp.eye(3)

    A_ion = (e / Mi) * K - src.nu_in * I3                      # (3, 3) constante
    A_ee = -(e / Me) * K - (src.nu_en + src.nu_ei) * I3        # (3, 3) constante
    ratio = ne / jnp.maximum(ni, DENS_FLOOR)                   # (...)
    A_ei = -src.nu_ei * ratio[..., None, None] * I3            # (..., 3, 3)

    shp = ni.shape
    A = jnp.zeros(shp + (6, 6))
    A = A.at[..., 0:3, 0:3].set(A_ion)
    A = A.at[..., 3:6, 3:6].set(A_ee)
    A = A.at[..., 3:6, 0:3].set(A_ei)

    g, U = src.g, src.U
    ni_ = ni[..., None]
    ne_ = ne[..., None]
    b_i = e * ni_ * g + (e * e * ni_ / Mi) * E + src.nu_in * e * ni_ * U
    b_e = -e * ne_ * g + (e * e * ne_ / Me) * E - src.nu_en * e * ne_ * U
    b = jnp.concatenate([b_i, b_e], axis=-1)        # (..., 6)

    j_old = jnp.concatenate([ji, je], axis=-1)      # (..., 6)
    I6 = jnp.eye(6)
    M = I6 - dt * A
    rhs = (j_old + dt * b)[..., None]               # (..., 6, 1)
    j_new = jnp.linalg.solve(M, rhs)[..., 0]        # (..., 6)

    new = arr.at[..., 2:8].set(j_new)
    return unstack_state(new)

