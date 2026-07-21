# `plume_si_epb.py` — Pluma EPB en unidades SI

Simulación **2D no lineal** de una **burbuja de plasma ecuatorial** (Equatorial
Plasma Bubble, EPB) en unidades físicas (SI), sobre un plano vertical
ecuatorial. Reproduce la figura clásica de la literatura EPB: depleciones del
*bottomside* de la capa F que ascienden como **plumas** hacia el *topside* por
la inestabilidad de **Rayleigh–Taylor colisional**.

- **x**: distancia zonal $[0, 400]$ km
- **z**: altitud $[200, 800]$ km
- **Salida**: `figures/fig5_pluma_si.png` (4 mapas de $\log_{10} n_e(x,z)$).

---

## 1. Cómo correrlo

Desde la raíz del repo, con el entorno virtual activo:

```bash
cd /home/est3ban3/PyExnerFork/PyExner
MPIR_CVAR_ENABLE_GPU=0 JAX_PLATFORMS=cpu python tests/runtime/EPB/plume_si_epb.py
```

Variables de entorno:
- `JAX_PLATFORMS=cpu` — fuerza JAX a CPU (el problema es chico, $768\times144$, y
  evita conflictos GPU/MPI).
- `MPIR_CVAR_ENABLE_GPU=0` — desactiva soporte GPU de MPICH (los kernels
  importan `mpi4jax`/`mpi4py`, aunque este script no distribuye).

No toma argumentos de línea de comandos: **todos los parámetros son constantes
al principio del archivo**. Para cambiar el escenario, se editan esas
constantes.

---

## 2. Parámetros de entrada (constantes del script)

### Dominio / malla
| Constante | Valor | Significado |
|---|---|---|
| `Z0, Z1` | `200e3, 800e3` | rango de altitud [m] |
| `LX` | `400e3` | extensión zonal [m] |
| `NX` | `768` | celdas en x |
| `DX` | `LX/NX` ≈ 4167 m | **malla isótropa** ($dx=dz$, requisito de los kernels) |
| `NZ` | `int((Z1-Z0)/DX)` = 144 | celdas en z |

### Fondo ionosférico (escenario post-PRE)
| Constante | Valor | Significado |
|---|---|---|
| `FR` | `FRegionParams(h_peak=450e3)` | capa F2 **elevada** ($h_mF2=450$ km) |
| `N_FLOOR` | `5e-3 · n_max` | piso suave de densidad (suaviza el *wrap* en z) |
| `H_REF` | `420e3` | altura de referencia (bottomside elevado) |
| `B0` | `dipole_B(H_REF)` ≈ 2.4e-5 T | campo magnético en H_REF |
| `G0` | `gravity(H_REF)` ≈ 8.6 m/s² | gravedad en H_REF |
| `NU_IN` | `0.05` | frecuencia colisión ion–neutro [s⁻¹] |
| `NU_EN` | `0.005` | frecuencia colisión electrón–neutro [s⁻¹] |

### Física y numérica
| Constante | Significado |
|---|---|
| `PHYS = EPBPhysParams(...)` | constantes físicas SI, **plasma frío** `Ti=Te=0` (elimina el CFL electrónico $c_e\sim120$ km/s) |
| `SRC = EPBSourceParams(gz=-G0, By=B0, nu_in, nu_en, poisson_iters=800)` | fuentes de momento + iteraciones del solve de Poisson |
| `N_HARD` | `1e-3·n_max`, piso duro post-transporte (positividad MUSCL) |
| `SIGMA_MIN` | piso de conductividad Pedersen (sustituto 2D de la conductancia integrada del tubo; limita $v_\text{burbuja}$) |
| `D_NUM` | `0.15·V_REF·dx` ≈ 1.1e5 m²/s, difusión sub-grid (mata modos de rejilla / Nyquist) |
| `V_REF` | `G0/NU_IN` ≈ 173 m/s, velocidad de deriva de referencia |

### Control temporal (dentro de `run()`)
| Variable | Valor | Significado |
|---|---|---|
| `CFL` | `0.2` | número de Courant de las derivas |
| `DT_MAX` | `6.0` s | paso máximo |
| `V_FLOOR` | `40.0` m/s | velocidad mínima para el cálculo de dt |
| `T_FINAL` | `2800.0` s | tiempo final (~47 min) |
| `SNAP_T` | `[0, 1400, 2100, 2800]` s | instantes de las 4 fotos |

---

## 3. Estructura y flujo del código

### 3.1 Setup (nivel de módulo)
1. **Imports y config JAX** (`jax_enable_x64=True` → float64, imprescindible en SI).
2. **Malla**: `zc`, `xc`, `X`, `Z` (meshgrid, forma `(NZ, NX)`).
3. **Fondo vertical**: perfil Chapman `n0_z`, recombinación `beta_z`, producción
   `PROD = beta·n0` (equilibrio fotoquímico que *ancla* el fondo), `BETA`.
4. **Condición inicial**: `n_init = n0_z · seed`, con `seed` una **semilla
   multi-modo** (cosenos en x, modos 2 y 3) modulada por una gaussiana centrada
   en el bottomside. Siembra la perturbación que la RT amplifica.
5. **Estado inicial** `STATE0`: `EPBTwoFluidState` con $n_i=n_e=n_\text{init}$ y
   todas las corrientes en cero.

### 3.2 Operadores numéricos locales (definidos en el script)
- `_mc_slope(n, axis)` — pendiente limitada **MC (monotonized central)**,
  periódica → reconstrucción MUSCL de 2º orden monótona.
- `_advect_div(n, vx, vz)` — divergencia conservativa $\nabla\cdot(n\mathbf{v})$
  vía **MUSCL upwind** en caras.
- `_lap(n)` — laplaciano 5 puntos periódico (difusión sub-grid).

### 3.3 `step(s, dt)` — un paso temporal (jiteado con `@jax.jit`)
Orden: **transporte → potencial → fuentes → química**:

1. **Derivas por celda**: $v = j/(q\,n)$ (signo: ion `+`, electrón `−`), con
   piso `N_HARD`.
2. **Transporte de continuidad** de $n_i$ y $n_e$ por separado (cada especie con
   su deriva) + difusión numérica. **Solo se advecta la continuidad**, no el
   momento (la corriente es diagnóstica; advectar su inercia explota cuando
   $n\to0$ en la deplecion).
3. **Conductividad Pedersen** $\sigma_P = \max(\sigma_P(n), \sigma_\text{min})$.
4. **Cierre electrostático drive-driven** (tipo Ossakow): corriente motriz
   gravitacional $J_g = n M_i g / B$ y su divergencia centrada como RHS, y se
   resuelve
   $$\nabla\cdot(\sigma_P \nabla\phi) = \nabla\cdot J_g$$
   con `_poisson_cg_sigma`. **Crítico**: el RHS es *solo* la corriente motriz,
   no la corriente total del estado (la total genera un lazo *lagged*
   inestable, ~×5/paso).
5. **Campo eléctrico** $E=-\nabla\phi$ (`electric_field`).
6. **Fuentes rígidas implícitas** (`implicit_source_solve` con `E_ext=E`):
   backward Euler 6×6 por celda que regenera las derivas estacionarias
   ($E\times B$ + Pedersen + gravitacional). A-estable, absorbe la rigidez de
   girofrecuencia $\Omega\,dt\gg1$.
7. **Química** (`chemistry_step_state`): $n^{k+1}=(n^k+dt\,P)/(1+dt\,\beta)$,
   backward Euler positivo.

### 3.4 `run()` — lazo temporal
- Recalcula `dt` cada 5 pasos con **CFL de las derivas** (`_vmax` da
  $\max|j|/(en)$), acotado por `DT_MAX`.
- Ajusta `dt_eff` para caer exacto en los `SNAP_T`.
- Guarda 4 snapshots de $n_e$.
- Imprime diagnósticos: profundidad de deplecion cada 40 pasos, y al final la
  **velocidad de ascenso del ápice** (comparada con 100–500 m/s observados) y la
  profundidad de deplecion final.

### 3.5 `plot(snaps)` — figura
4 paneles `pcolormesh` de $\log_{10}n_e$, con línea del $h_mF2$, colorbar y
título, guardados en `figures/fig5_pluma_si.png`.

---

## 4. Funciones que usa de PyExner

### Kernels EPB (lo "nuevo")

**`PyExner.solvers.kernels.epb_twofluid`**
- `EPBPhysParams` — constantes físicas del cierre (e, kB, Mi, Me, Ti, Te). Aquí
  con `Ti=Te=0` (frío).

**`PyExner.solvers.kernels.epb_sources`**
- `EPBSourceParams` — campos de fondo (g, B, U, E₀), frecuencias de colisión y
  `poisson_iters`.
- `implicit_source_solve(...)` — **backward Euler 6×6** de las fuentes rígidas;
  se llama con `E_ext=E` (campo congelado, el caller maneja el cierre
  electrostático).
- `electric_field(phi, src, dx)` — $E = E_0 - \nabla\phi$ (gradiente forward,
  consistente con el solve).
- `_poisson_cg_sigma(rhs, sigma, dx, n_iter)` — CG para $-L_\sigma\phi=-\text{rhs}$
  con **coeficiente variable** $\sigma$ en caras (SPD, media cero).

**`PyExner.solvers.kernels.epb_fluxtube`**
- `pedersen_conductivity(n, nu, B)` — $\sigma_P=(ne/B)\,\nu\Omega/(\nu^2+\Omega^2)$
  [S/m].

**`PyExner.solvers.kernels.epb_ionosphere`**
- `SI_CONST` — constantes SI (e, kB, amu, Me, RE).
- `M_OPLUS` — masa del O⁺.
- `FRegionParams` — parámetros de la región F ecuatorial (n_max, h_peak, escalas…).
- `chapman_layer(z, ...)` — perfil de densidad Chapman-α (bottomside afilado).
- `exp_profile(z, ...)` — perfil exponencial (para $\beta(z)$).
- `dipole_B(z, ...)` — campo dipolar $B(z)=B_\text{surf}(RE/(RE+z))^3$.
- `gravity(z, ...)` — $g(z)=g_\text{surf}(RE/(RE+z))^2$.
- `chemistry_step_state(state, dt, prod, beta)` — paso químico implícito sobre el
  estado.

### Estado (rama EPB)

**`PyExner.state.epb_twofluid_state`**
- `EPBTwoFluidState` — dataclass del vector conservado
  $Q=[n_i,n_e,j_{ix},j_{iy},j_{iz},j_{ex},j_{ey},j_{ez}]$, con método `.replace(...)`.

### Infraestructura base de PyExner (indirecta)
El script **no** usa directamente el driver, integradores, `roe_solver`, `mesh`
ni I/O: implementa su propio lazo temporal y su propio transporte MUSCL. Hereda
del núcleo:
- El **registro de estados** (`@register_state("EPB_TwoFluid")`), que integra
  este estado al framework general.
- La convención de paquete/imports y `EPB_FIELD_ORDER` como única fuente de
  verdad del layout, compartida con kernels de transporte, I/O y registros.
- Dependencias base: `jax`, `mpi4jax`/`mpi4py` (importadas por `epb_twofluid`,
  aunque este run es serial).

---

## 5. Idea física en una línea

Plasma **frío** (sin CFL acústico) + cierre **electrostático diagnóstico**
($\phi$ de la corriente motriz gravitacional) + **derivas implícitas**
A-estables ⟹ el $dt$ lo fija solo el CFL de las derivas (~2–6 s), y la RT
colisional ($\gamma\sim g/(\nu L_n)$) hace crecer las plumas de forma estable
hasta que penetran el *topside*.
