"""Estado conservado del modelo EPB de dos fluidos (rama ``EPB_TwoFluid``).

Modelo 2.5D de Burbujas de Plasma Ecuatoriales con formulacion hiperbolica de
dos fluidos. El vector de estado conservado, en el orden FISICO exacto que debe
respetarse en kernels, registros, I/O y pruebas, es:

    Q = [ n_i, n_e, j_ix, j_iy, j_iz, j_ex, j_ey, j_ez ]^T

donde:
    n_i, n_e          densidades de numero de iones y electrones
    j_i{x,y,z}        componentes del flujo de momento ionico
    j_e{x,y,z}        componentes del flujo de momento electronico

Geometria: plano discretizado (x, z) (2.5D); las tres componentes de corriente
se transportan, pero solo x y z se discretizan espacialmente.

Cierre: isotermo, p_alpha = n_alpha k_B T_alpha (la fisica de presion y fuentes
se introduce en fases posteriores; este estado solo almacena las variables
conservadas).
"""

from dataclasses import dataclass, replace as dc_replace

import jax
import jax.numpy as jnp

from PyExner.state.registry import register_state


# Orden canonico de las componentes conservadas de Q. Es la unica fuente de
# verdad del layout; kernels, I/O y pruebas deben referenciar este orden.
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
        """Materializa todas las componentes en host (NumPy) para I/O.

        El estado es un pytree de JAX, por lo que ``tree_map`` recorre las 8
        componentes conservadas en el orden canonico.
        """
        return jax.tree_util.tree_map(jax.device_get, self)


def EPBTwoFluid_state_flatten(state: EPBTwoFluidState):
    children = tuple(getattr(state, name) for name in EPB_FIELD_ORDER)
    return children, None


def EPBTwoFluid_state_unflatten(aux, children):
    return EPBTwoFluidState(*children)


jax.tree_util.register_pytree_node(
    EPBTwoFluidState, EPBTwoFluid_state_flatten, EPBTwoFluid_state_unflatten
)
