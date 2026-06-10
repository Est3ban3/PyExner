import os
import jax
import jax.numpy as jnp
from mpi4py import MPI
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dev = jax.devices()[0]
x = jnp.ones(4, dtype=jnp.float32) * (rank + 1)

# device_buffer location
plat = x.device.platform if hasattr(x, "device") else "?"

# Allreduce sum over GPU-resident arrays
res = mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)
y = res[0] if isinstance(res, tuple) else res
y.block_until_ready()

expected = sum(range(1, size + 1))
ok = bool(jnp.all(y == expected))
if rank == 0:
    print(f"[rank {rank}/{size}] backend={plat} device={dev}")
    print(f"allreduce result = {y[0]} (expected {expected}) -> {'OK' if ok else 'FAIL'}")
