# PyExner/state/epb_twofluid_state.py
"""Estado conservado del modelo de dos fluidos (rama EPB_TwoFluid).

El vector conservado, en el orden que usa TODO el resto (kernels, registros,
I/O, pruebas), es:

    Q = [ n_i, n_e, j_ix, j_iy, j_iz, j_ex, j_ey, j_ez ]

densidades de numero y densidades de corriente por especie. Si alguna vez se
cambia este orden hay que tocar EPB_FIELD_ORDER y nada mas: es la unica
fuente de verdad del layout.

Geometria 2.5D: se discretiza el plano (x, z) pero las corrientes llevan sus
tres componentes (la y entra por j x B).
"""

from dataclasses import dataclass, replace as dc_replace

import jax
import jax.numpy as jnp

from PyExner.state.registry import register_state


# Orden canonico de las componentes de Q. No reordenar sin revisar kernels e I/O.
EPB_FIELD_ORDER = (
    "n_i",
    "n_e",
    "j_ix",
    "j_iy",
    "j_iz",
    "j_ex",
    "j_ey",
    "j_ez",
)


@register_state("EPB_TwoFluid")
@dataclass
class EPBTwoFluidState:
    n_i: jax.Array
    n_e: jax.Array
    j_ix: jax.Array
    j_iy: jax.Array
    j_iz: jax.Array
    j_ex: jax.Array
    j_ey: jax.Array
    j_ez: jax.Array

    @classmethod
    def empty(cls, mesh: "Mesh2D", dtype=jnp.float32) -> "EPBTwoFluidState":
        shape = mesh.local_shape
        zeros = jnp.zeros(shape, dtype=dtype)
        return cls(
            n_i=zeros,
            n_e=zeros,
            j_ix=zeros,
            j_iy=zeros,
            j_iz=zeros,
            j_ex=zeros,
            j_ey=zeros,
            j_ez=zeros,
        )

    @classmethod
    def from_params(cls, params: dict, dtype=jnp.float32) -> "EPBTwoFluidState":
        def field(name):
            val = params.get(f"{name}_init")
            return jnp.asarray(val, dtype=dtype)

        return cls(**{name: field(name) for name in EPB_FIELD_ORDER})

    def replace(self, **kwargs) -> "EPBTwoFluidState":
        return dc_replace(self, **kwargs)

    def to_host(self) -> "EPBTwoFluidState":
        """Baja las 8 componentes a NumPy (host) para el I/O paralelo."""
        return jax.tree_util.tree_map(jax.device_get, self)


def EPBTwoFluid_state_flatten(state: EPBTwoFluidState):
    children = tuple(getattr(state, name) for name in EPB_FIELD_ORDER)
    return children, None


def EPBTwoFluid_state_unflatten(aux, children):
    return EPBTwoFluidState(*children)


jax.tree_util.register_pytree_node(
    EPBTwoFluidState, EPBTwoFluid_state_flatten, EPBTwoFluid_state_unflatten
)
