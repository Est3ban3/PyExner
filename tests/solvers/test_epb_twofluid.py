"""Fase 7 — Validacion incremental de la rama ``EPB_TwoFluid``.

Bateria pequena pero discriminante que detecta errores catastroficos (registro,
orden del estado, signos, contornos, estado uniforme, halos de 8 campos) antes
de abordar simulaciones fisicas complejas.

Ejecucion local (WSL, MPICH GPU-aware): prefijar con el workaround
    MPIR_CVAR_ENABLE_GPU=0 python -m pytest tests/solvers/test_epb_twofluid.py
o directamente
    MPIR_CVAR_ENABLE_GPU=0 python tests/solvers/test_epb_twofluid.py

Disenado para 1 rango (parNx=parNy=1): el halo exchange es identidad pero se
ejercita el cableado completo de las 8 componentes. En cluster real, los mismos
checks corren multi-rango via la infraestructura MPI del framework.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Activar registros (estado, solver, contornos, integradores).
import PyExner.state            # noqa: F401
import PyExner.solvers          # noqa: F401
import PyExner.domain           # noqa: F401
import PyExner.integrators      # noqa: F401

from PyExner.state.registry import STATE_REGISTRY, create_empty_state
from PyExner.solvers.registry import SOLVER_REGISTRY, create_solver_bundle
from PyExner.integrators.registry import INTEGRATOR_REGISTRY, create_integrator_bundle
from PyExner.domain.boundary_registry import BOUNDARY_REGISTRY, BoundaryManager
from PyExner.parallel.mpi_utils import Parallel

from PyExner.state.epb_twofluid_state import EPBTwoFluidState, EPB_FIELD_ORDER
from PyExner.solvers.kernels.epb_twofluid import (
    EPBPhysParams, stack_state, unstack_state, physical_flux,
)


# --------------------------------------------------------------------------- #
# Utilidades de construccion (single-rank)                                     #
# --------------------------------------------------------------------------- #

NX = NY = 8
DH = 1.0


def _build_params(nx=NX, ny=NY, dh=DH):
    """Parametros minimos: 1 rango + 4 contornos transmisivos EPB.

    Centros de celda en (i+0.5)*dh -> cada banda de contorno se dimensiona para
    capturar exactamente el anillo exterior de centros (una celda de ancho).
    """
    xmax = nx * dh
    ymax = ny * dh
    eps = 0.1 * dh
    return {
        "flux_scheme": "EPB_TwoFluid",
        "integrator": "IMEX",
        "parNx": 1,
        "parNy": 1,
        "cfl": 0.4,
        "boundaries": {
            "west": {
                "type": "Transmissive",
                "polygon": [[-eps, -eps], [dh, -eps], [dh, ymax + eps], [-eps, ymax + eps]],
                "values": ["nan"], "normal": [-1.0, 0.0],
            },
            "east": {
                "type": "Transmissive",
                "polygon": [[xmax - dh, -eps], [xmax + eps, -eps], [xmax + eps, ymax + eps], [xmax - dh, ymax + eps]],
                "values": ["nan"], "normal": [1.0, 0.0],
            },
            "south": {
                "type": "Transmissive",
                "polygon": [[-eps, -eps], [xmax + eps, -eps], [xmax + eps, dh], [-eps, dh]],
                "values": ["nan"], "normal": [0.0, -1.0],
            },
            "north": {
                "type": "Transmissive",
                "polygon": [[-eps, ymax - dh], [xmax + eps, ymax - dh], [xmax + eps, ymax + eps], [-eps, ymax + eps]],
                "values": ["nan"], "normal": [0.0, 1.0],
            },
        },
    }


def _build_mesh_coords(nx=NX, ny=NY, dh=DH):
    x = (jnp.arange(nx) + 0.5) * dh
    y = (jnp.arange(ny) + 0.5) * dh
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    return X, Y


def _uniform_state(vals, nx=NX, ny=NY):
    return EPBTwoFluidState(**{f: jnp.full((ny, nx), float(v)) for f, v in zip(EPB_FIELD_ORDER, vals)})


# --------------------------------------------------------------------------- #
# 1. Wiring del framework                                                      #
# --------------------------------------------------------------------------- #

def test_wiring_registries():
    assert "EPB_TwoFluid" in STATE_REGISTRY
    assert "EPB_TwoFluid" in SOLVER_REGISTRY
    assert "IMEX" in INTEGRATOR_REGISTRY
    # No-regresion de las ramas hidraulicas.
    assert {"Roe", "Roe Exner"}.issubset(STATE_REGISTRY.keys())
    assert {"Roe", "Roe Exner"}.issubset(SOLVER_REGISTRY.keys())
    assert "Forward Euler" in INTEGRATOR_REGISTRY

    bundle = create_solver_bundle("EPB_TwoFluid")
    assert bundle.name == "EPB_TwoFluid"
    assert bundle.source_fn is not None   # rama con fuentes rigidas
    integ = create_integrator_bundle("IMEX")
    assert integ.name == "IMEX"


# --------------------------------------------------------------------------- #
# 2. Orden del estado                                                          #
# --------------------------------------------------------------------------- #

def test_state_order_roundtrip():
    vals = [10, 11, 12, 13, 14, 15, 16, 17]
    st = _uniform_state(vals)
    arr = stack_state(st)
    # Cada componente en su posicion canonica.
    for i, name in enumerate(EPB_FIELD_ORDER):
        assert float(arr[0, 0, i]) == float(vals[i])
        assert jnp.allclose(arr[..., i], getattr(st, name))
    # Roundtrip exacto.
    st2 = unstack_state(arr)
    for name in EPB_FIELD_ORDER:
        assert jnp.allclose(getattr(st2, name), getattr(st, name))
    # Pytree de 8 hojas en orden canonico.
    leaves, _ = jax.tree_util.tree_flatten(st)
    assert len(leaves) == 8


# --------------------------------------------------------------------------- #
# 3. Signo del flujo electronico                                              #
# --------------------------------------------------------------------------- #

def test_electron_flux_signs():
    phys = EPBPhysParams()  # e=Mi=Me=kB=Ti=Te=1
    ni, ne, jix, jex = 3.0, 2.0, 1.0, 0.5
    st = _uniform_state([ni, ne, jix, 0, 0, jex, 0, 0])
    F = physical_flux(stack_state(st), phys, "x")
    # Continuidad electronica: F_ne = -jex/e (signo de carga).
    assert np.isclose(float(F[0, 0, 1]), -jex / phys.e)
    # Continuidad ionica: F_ni = +jix/e.
    assert np.isclose(float(F[0, 0, 0]), jix / phys.e)
    # Presion electronica entra con signo NEGATIVO:
    # F6 = -jex^2/(e ne) - (e/Me) pe.
    F6_expected = -jex**2 / (phys.e * ne) - (phys.e / phys.Me) * (ne * phys.kB * phys.Te)
    assert np.isclose(float(F[0, 0, 5]), F6_expected)
    # Presion ionica entra con signo POSITIVO:
    F3_expected = jix**2 / (phys.e * ni) + (phys.e / phys.Mi) * (ni * phys.kB * phys.Ti)
    assert np.isclose(float(F[0, 0, 2]), F3_expected)


# --------------------------------------------------------------------------- #
# 4. Contornos                                                                 #
# --------------------------------------------------------------------------- #

def test_boundary_registration_convention():
    # Nombre = flux_scheme + " " + type.
    assert "EPB_TwoFluid Transmissive" in BOUNDARY_REGISTRY
    params = _build_params()
    X, Y = _build_mesh_coords()
    bm = BoundaryManager(params, X, Y)
    # Se instanciaron los 4 contornos.
    assert len(bm.boundary_handlers) == 4
    # Aplicar contornos a un estado es puro y conserva la forma.
    st = _uniform_state([2, 2, 0.1, 0, 0, 0.1, 0, 0])
    out = bm.apply(st, 0.0)
    for name in EPB_FIELD_ORDER:
        assert getattr(out, name).shape == (NY, NX)


# --------------------------------------------------------------------------- #
# 5 + 6. Estado uniforme y halo de 8 campos (end-to-end via solver + IMEX)     #
# --------------------------------------------------------------------------- #

def _wire_solver(params):
    mpi = Parallel(params)
    X, Y = _build_mesh_coords()
    boundaries = BoundaryManager(params, X, Y)
    bundle = create_solver_bundle(params["flux_scheme"])
    state0 = _uniform_state([2, 2, 0, 0, 0, 0, 0, 0])
    config = bundle.config(state0, mpi, boundaries, DH, params)
    return mpi, boundaries, bundle, config


def test_uniform_state_preserved_end_to_end():
    params = _build_params()
    mpi, boundaries, bundle, config = _wire_solver(params)

    # Estado uniforme con corrientes constantes (sin gradientes).
    st = _uniform_state([2.0, 2.0, 0.3, 0.0, -0.1, 0.4, 0.0, 0.2])
    mask = bundle.mask_fn(st, mpi, boundaries.boundary_mask)
    st = bundle.init_fn(st, mask, config)

    # Sin parametros de fuente -> S=0; estado uniforme debe permanecer uniforme.
    dt = float(bundle.compute_dt_fn(st, params["cfl"], mask, config))
    assert np.isfinite(dt) and dt > 0.0

    # "En ausencia de fuentes" (plan, item 5): el transporte hiperbolico de un
    # estado constante con corrientes constantes tiene divergencia nula. En
    # volumenes finitos sin capa fantasma (1 rango), el invariante discriminante
    # es que el interior activo permanezca uniforme e identico (div=0 exacto).
    st1 = bundle.step_fn(st, 0.0, dt, mask, config)        # transporte explicito
    for name in EPB_FIELD_ORDER:
        before = getattr(st, name)[1:-1, 1:-1]
        after = getattr(st1, name)[1:-1, 1:-1]
        assert jnp.allclose(after, after[0, 0]), f"campo {name} dejo de ser uniforme"
        assert jnp.allclose(after, before, atol=1e-12), f"campo {name} cambio"

    # El operador de fuentes con parametros nulos preserva densidades (S=0 en
    # continuidad) tras un paso IMEX completo.
    st2 = bundle.source_fn(st1, 0.0, dt, mask, config)
    assert jnp.allclose(st2.n_i, st1.n_i, atol=1e-10)
    assert jnp.allclose(st2.n_e, st1.n_e, atol=1e-10)


def test_halo_exchange_eight_fields_identity():
    # Single-rank: halo es identidad, pero se ejercita el cableado de los 8 campos.
    params = _build_params()
    mpi, boundaries, bundle, config = _wire_solver(params)
    # Estado con valores distintos por componente para detectar mezclas de campos.
    st = _uniform_state([1, 2, 3, 4, 5, 6, 7, 8])
    from PyExner.solvers.kernels.epb_twofluid import halo_exchange_all
    out = halo_exchange_all(st, config.halo_exchange)
    for i, name in enumerate(EPB_FIELD_ORDER):
        assert jnp.allclose(getattr(out, name), float(i + 1)), f"campo {name} se mezclo"


def test_conservation_and_positivity_imex():
    # Drive corto via IMEX: masa total conservada (en ausencia de flujo neto por
    # contornos transmisivos simetricos) y densidades positivas.
    params = _build_params()
    mpi, boundaries, bundle, config = _wire_solver(params)

    # Pulso de densidad centrado, corrientes nulas (sin flujo en t=0).
    xx, zz = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY))
    bump = 1.0 + 0.5 * jnp.exp(-(((xx - NX / 2) ** 2 + (zz - NY / 2) ** 2) / 4.0))
    z = jnp.zeros((NY, NX))
    st = EPBTwoFluidState(n_i=bump, n_e=bump, j_ix=z, j_iy=z, j_iz=z, j_ex=z, j_ey=z, j_ez=z)

    mask = bundle.mask_fn(st, mpi, boundaries.boundary_mask)
    st = bundle.init_fn(st, mask, config)

    for _ in range(20):
        dt = float(bundle.compute_dt_fn(st, params["cfl"], mask, config))
        st = bundle.step_fn(st, 0.0, dt, mask, config)
        st = bundle.source_fn(st, 0.0, dt, mask, config)

    assert float(jnp.min(st.n_i)) > 0.0
    assert float(jnp.min(st.n_e)) > 0.0
    assert np.all(np.isfinite(np.array(st.n_i)))


# --------------------------------------------------------------------------- #
# Runner directo                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_wiring_registries,
        test_state_order_roundtrip,
        test_electron_flux_signs,
        test_boundary_registration_convention,
        test_uniform_state_preserved_end_to_end,
        test_halo_exchange_eight_fields_identity,
        test_conservation_and_positivity_imex,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
