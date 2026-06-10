# PyExner — Documentación de Arquitectura

## 1. Visión General

**PyExner** es un solver 2D de alto rendimiento para las ecuaciones de aguas someras (Shallow Water Equations, SWE) acopladas con la ecuación de Exner (transporte de sedimentos y evolución morfodinámica del lecho). Está construido sobre **JAX** para aceleración en GPU y utiliza **MPI** (via `mpi4py` y `mpi4jax`) para computación distribuida en múltiples GPUs/nodos.

### Ecuaciones que resuelve

1. **Ecuaciones de aguas someras (SWE):** Conservación de masa y momento en flujo de agua poco profunda.
   - $\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = 0$
   - $\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\left(hu^2 + \frac{1}{2}gh^2\right) + \frac{\partial (huv)}{\partial y} = -gh\frac{\partial z}{\partial x} - g h S_{f,x}$
   - $\frac{\partial (hv)}{\partial t} + \frac{\partial (huv)}{\partial x} + \frac{\partial}{\partial y}\left(hv^2 + \frac{1}{2}gh^2\right) = -gh\frac{\partial z}{\partial y} - g h S_{f,y}$

2. **Ecuación de Exner:** Evolución del lecho por transporte de sedimentos.
   - $\frac{\partial z_b}{\partial t} + \frac{1}{1-p}\nabla \cdot \mathbf{q_b} = 0$

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| Cómputo numérico | JAX (`jax`, `jax.numpy`) |
| Paralelismo distribuido | MPI (`mpi4py`, `mpi4jax`) |
| I/O paralelo | PnetCDF (`pnetcdf`) |
| Configuración | YAML (`pyyaml`) |
| Visualización | Matplotlib, xarray |

---

## 2. Punto de Entrada — `run_driver()`

**Archivo:** [src/PyExner/runtime/driver.py](src/PyExner/runtime/driver.py)

Esta es la función principal que orquesta toda la simulación. El flujo es:

```
config YAML → Parallel (MPI) → PnetCDF I/O → Mesh → State → Boundaries → Solver → Integrator → Ejecución
```

### Pasos detallados:

1. **Lee el archivo YAML** de configuración (`config_path`).
2. **Inicializa MPI** via `Parallel(params)` — crea la topología cartesiana.
3. **Crea el lector PnetCDF** — abre archivos de entrada/salida en paralelo.
4. **Genera la malla** (`pnetcdf_io.generate_mesh()`) — divide el dominio entre procesos MPI.
5. **Crea el estado vacío** según el esquema de flujo (`"Roe"` o `"Roe Exner"`).
6. **Lee las condiciones iniciales** del archivo NetCDF al estado.
7. **Configura condiciones de contorno** (`BoundaryManager`).
8. **Selecciona el solver** del registro (`SOLVER_REGISTRY`).
9. **Selecciona el integrador temporal** del registro (`INTEGRATOR_REGISTRY`).
10. **Ejecuta el bucle temporal** (`integrator.run_fn()`).

### Parámetros YAML clave:

| Parámetro | Descripción | Ejemplo |
|---|---|---|
| `end_time` | Tiempo final de simulación | `20.0` |
| `out_freq` | Frecuencia de escritura de salida | `0.25` |
| `cfl` | Número CFL para control de paso de tiempo | `0.5` |
| `flux_scheme` | Esquema numérico: `"Roe"` o `"Roe Exner"` | `"Roe Exner"` |
| `integrator` | Esquema temporal: `"Forward Euler"` o `"SSPRK2"` | `"Forward Euler"` |
| `parNx`, `parNy` | Particiones MPI en X e Y | `2`, `2` |
| `input_file` | Archivo NetCDF de entrada | `"input.nc"` |
| `output_file` | Archivo NetCDF de salida | `"output.nc"` |
| `boundaries` | Definición de condiciones de contorno | (ver sección 6) |
| `erosion` | Parámetros de sedimentos (solo `Roe Exner`) | (ver sección 5) |

---

## 3. Módulo `state/` — Variables de Estado

**Directorio:** [src/PyExner/state/](src/PyExner/state/)

Define las variables conservativas que el solver transporta en cada celda de la malla.

### `BaseState` — [state/base.py](src/PyExner/state/base.py)

Clase base con los campos hidráulicos comunes:

| Campo | Tipo | Descripción |
|---|---|---|
| `h` | `jax.Array` | Profundidad del agua |
| `hu` | `jax.Array` | Momento en X ($h \cdot u$) |
| `hv` | `jax.Array` | Momento en Y ($h \cdot v$) |

Métodos utilitarios: `unshard()`, `__getitem__()`, `reshape()`, `apply_to_all()`, `to_host()`.

### `RoeState` — [state/roe_state.py](src/PyExner/state/roe_state.py)

Para simulaciones **solo hidráulicas** (SWE):

| Campo adicional | Descripción |
|---|---|
| `z` | Elevación del lecho (fija) |
| `n` | Coeficiente de rugosidad de Manning |

Métodos: `empty(mesh)`, `from_params(params)`, `replace(**kwargs)`.

### `RoeExnerState` — [state/roe_exner_state.py](src/PyExner/state/roe_exner_state.py)

Para simulaciones **hidrodinámicas + morfodinámicas** (SWE + Exner):

| Campo adicional | Descripción |
|---|---|
| `z` | Elevación del lecho no-erodible (roca) |
| `z_b` | Elevación del lecho erodible (sedimento) |
| `n` | Coeficiente de rugosidad de Manning (original) |
| `n_b` | Rugosidad efectiva (calculada según $d_{50}$ del sedimento) |
| `G` | Factor de interacción morfodinámica (tasa de transporte de sedimento) |
| `seds` | Array con propiedades de sedimentos (fracción, diámetro, densidad, etc.) |

### Registro de estados — [state/registry.py](src/PyExner/state/registry.py)

Sistema de registro con decorador:
- `@register_state("Roe")` → registra `RoeState`
- `@register_state("Roe Exner")` → registra `RoeExnerState`
- `create_empty_state(name, mesh, rank)` → crea instancia vacía según el nombre.
- `create_state(name, params)` → crea instancia desde parámetros.

---

## 4. Módulo `solvers/` — Solvers de Riemann

**Directorio:** [src/PyExner/solvers/](src/PyExner/solvers/)

### 4.1 Arquitectura del Registro

**Archivo:** [solvers/registry.py](src/PyExner/solvers/registry.py)

Cada solver se registra como un `SolverBundle` (NamedTuple) con:

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | `str` | Nombre del solver |
| `config` | `Callable` | Función de configuración (crea `SolverConfig`) |
| `mask_fn` | `Callable` | Genera máscaras de celdas activas/inactivas |
| `init_fn` | `Callable` | Inicializa el estado (intercambio de halos) |
| `step_fn` | `Callable` | **Avanza un paso temporal** |
| `compute_dt_fn` | `Callable` | Calcula $\Delta t$ según condición CFL |

`SolverConfig` contiene: `mpi_handler`, `boundaries`, `dx`, `halo_exchange`.

### 4.2 Solver de Roe (SWE puro) — `"Roe"`

**Archivo del solver:** [solvers/roe_solver.py](src/PyExner/solvers/roe_solver.py)
**Kernels numéricos:** [solvers/kernels/roe.py](src/PyExner/solvers/kernels/roe.py)

Resuelve las ecuaciones de aguas someras 2D sin evolución del lecho.

#### Función principal: `roe_solver(si, sj, nx, ny, dx)`

Implementa el **solver de Riemann aproximado de Roe** en la interfaz entre dos celdas vecinas `si` y `sj`:

1. **Rotación al marco normal-tangencial:** Las velocidades se proyectan en componentes normal ($\hat{u}$) y tangencial ($\hat{v}$) a la interfaz.

2. **Promedios de Roe:**
   - $\tilde{h} = \frac{1}{2}(h_i + h_j)$
   - $\tilde{u} = \frac{\sqrt{h_i} \hat{u}_i + \sqrt{h_j} \hat{u}_j}{\sqrt{h_i} + \sqrt{h_j}}$
   - $\tilde{c} = \sqrt{g \tilde{h}}$

3. **Autovalores (velocidades de onda):**
   - $\lambda_1 = \tilde{u} - \tilde{c}$ (onda hacia la izquierda)
   - $\lambda_2 = \tilde{u}$ (onda de contacto)
   - $\lambda_3 = \tilde{u} + \tilde{c}$ (onda hacia la derecha)

4. **Corrección de entropía de Harten-Hyman:** Evita problemas numéricos en ondas de rarefacción transónicas.

5. **Autovectores:** Matriz $P$ y su inversa $P^{-1}$ para la descomposición modal.

6. **Ondas (alphas):** $\boldsymbol{\alpha} = P^{-1} \Delta U$ donde $\Delta U = U_j - U_i$.

7. **Términos fuente well-balanced:**
   - **Thrust topográfico:** Formulación especial para mantener equilibrio en reposo sobre batimetría variable (basado en [Murillo & García-Navarro, 2010](http://dx.doi.org/10.1016/j.jcp.2010.02.016)).
   - **Fricción de Manning:** $\tau = g \tilde{h} S_f \Delta x$ donde $S_f = \frac{n^2 \sqrt{u^2+v^2} \cdot u}{h^{4/3}}$.

8. **Upwinding:** Las contribuciones positivas (`upwP`) van a la celda derecha, las negativas (`upwM`) a la izquierda.

9. **Rotación inversa:** Se vuelve al marco global $(x, y)$.

#### `roe_solve_2D(state, dt, dx)` — Paso 2D completo

- Crea slices de interfaces en X e Y.
- Llama a `roe_solver()` en ambas direcciones.
- Acumula flujos y actualiza: $U^{n+1} = U^n - \frac{\Delta t}{\Delta x} F$.

#### `compute_dt(state, mask, dx)` — Cálculo del paso de tiempo

$$\Delta t = \text{CFL} \times \min\left(\frac{\Delta x}{|u| + c}, \frac{\Delta y}{|v| + c}\right)$$

Reducido globalmente via `MPI.MIN`.

#### Flujo del `step_fn_roe`:

```
Estado → roe_solve_2D → Intercambio de halos (MPI) 
→ Aplicar contorno → Estado actualizado
```
Estado → roe_solve_2D → Intercambio de halos (MPI) → Aplicar contorno → Estado actualizado
### 4.3 Solver de Roe-Exner (SWE + Exner) — `"Roe Exner"`

**Archivo del solver:** [solvers/roe_exner_solver.py](src/PyExner/solvers/roe_exner_solver.py)
**Kernels numéricos:** [solvers/kernels/roe_exner.py](src/PyExner/solvers/kernels/roe_exner.py)

Resuelve las ecuaciones de aguas someras **acopladas** con la ecuación de Exner para evolución del lecho.

#### Diferencias clave respecto al Roe puro:

1. **Estado extendido:** Incluye `z_b` (lecho erodible), `G` (factor de transporte), `n_b` (rugosidad por sedimento), `seds` (propiedades de sedimentos).

2. **Factor de interacción morfodinámica `G`:**
   La función `compute_G()` calcula la tasa de transporte de sedimento usando la **fórmula de Meyer-Peter & Müller (MPM)** modificada:

   - **Tensión de Shields:** $\theta = \frac{n_p^2 (u^2 + v^2)}{(s-1) d_{50} h^{1/3}}$
   - **Exceso de tensión:** $\Delta\theta = \max(\theta - \theta_c, 0)$
   - **Transporte tipo MPM:** $q_b \propto \gamma_1 \cdot \gamma_2 \cdot \gamma_3$
   - Soporta **múltiples fracciones de sedimento** con mezcla ponderada.

3. **Solver de Exner (`exner_solver`):**
   Implementa el **Approximately Coupled Method (ACM)** basado en [Martínez-Aranda et al., 2021](https://doi.org/10.1016/j.advwatres.2021.103931):
   - Calcula flujo de sedimento en interfaces usando upwinding basado en $\lambda_4$ (velocidad de onda de Exner).
   - Actualiza la elevación del lecho: $z_b^{n+1} = z_b^n - \frac{\Delta t}{\Delta x} \nabla \cdot \mathbf{q_b}$.

4. **Cálculo de velocidades de onda Exner (`_get_lambda`):**
   Resuelve el polinomio cúbico del sistema acoplado SWE-Exner usando las **fórmulas de Cardano-Vieta** (solución trigonométrica) para obtener los autovalores $\lambda_\text{min}$ y $\lambda_\text{max}$.

5. **Correcciones de momento (`momentum_corrections`):**
   Anula el momento en direcciones donde el agua no puede fluir (celdas secas adyacentes con lecho más alto que la superficie libre, o paredes).

#### Flujo del `step_fn_roeexner` (7 pasos):

```
1. roe_solve_2D         → Resolver hidrodinámica (SWE)
2. momentum_corrections → Corregir momento en celdas bloqueadas
3. boundaries.apply     → Aplicar condiciones de contorno
4. halo_exchange        → Sincronizar h, hu, hv entre procesos MPI
5. compute_G + halo     → Calcular y sincronizar factor de transporte G
6. exner_solve_2D       → Resolver evolución del lecho (Exner)
7. halo_exchange z_b    → Sincronizar lecho + recalcular rugosidad n_b
```

### 4.4 HLLC (Reservado)

**Archivo:** [solvers/kernels/hllc.py](src/PyExner/solvers/kernels/hllc.py) — Vacío, reservado para futura implementación.

---

## 5. Módulo `integrators/` — Integradores Temporales

**Directorio:** [src/PyExner/integrators/](src/PyExner/integrators/)

### Registro — [integrators/registry.py](src/PyExner/integrators/registry.py)

Cada integrador es un `IntegratorBundle`:

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | `str` | Nombre del integrador |
| `config` | `Callable` | Crea `IntegratorConfig` |
| `run_fn` | `Callable` | **Bucle temporal principal** |

`IntegratorConfig` contiene: `cfl`, `end_time`, `out_freq`, `solver_config`, `solver_bundle`.

### 5.1 Forward Euler — `"Forward Euler"`

**Archivo:** [integrators/forwardeuler.py](src/PyExner/integrators/forwardeuler.py)

Esquema de primer orden en tiempo. Estructura del bucle:

1. **Bucle exterior** (Python `while`): Controla escritura de salida y manejo de tiempos de output.
2. **Bucle interior** (`jax.lax.while_loop`): Iteraciones JIT-compiladas hasta el siguiente tiempo de output.
   - `cond_fn`: ¿El tiempo actual + dt supera el siguiente output? Si sí, sale del bucle.
   - `body_fn`: Calcula $\Delta t$ (CFL) → ejecuta `solver.step_fn()` → avanza el tiempo.
3. **Paso final**: Ajusta $\Delta t$ para aterrizar exactamente en el tiempo de output.
4. **Escritura**: `io.write_state()` en formato PnetCDF paralelo.

### 5.2 SSPRK2 — `"SSPRK2"`

**Archivo:** [integrators/SSPRK2.py](src/PyExner/integrators/SSPRK2.py)

Esquema Strong Stability Preserving Runge-Kutta de segundo orden:

$$U^{(1)} = U^n + \Delta t \cdot L(U^n)$$
$$U^{n+1} = \frac{1}{2} U^n + \frac{1}{2} U^{(1)} + \frac{1}{2} \Delta t \cdot L(U^{(1)})$$

Equivale a un promedio del estado original con dos pasos de Euler.

---

## 6. Módulo `domain/` — Malla y Condiciones de Contorno

**Directorio:** [src/PyExner/domain/](src/PyExner/domain/)

### 6.1 Malla — [domain/mesh.py](src/PyExner/domain/mesh.py)

`Mesh2D` es un dataclass con malla estructurada uniforme:

| Campo | Descripción |
|---|---|
| `global_Ny`, `global_Nx` | Dimensiones globales del dominio |
| `local_Ny`, `local_Nx` | Dimensiones locales (por proceso MPI) |
| `x_offset`, `y_offset` | Desplazamiento del subdominio local |
| `local_X`, `local_Y` | Matrices de coordenadas locales |
| `dh` | Espaciamiento de la malla ($\Delta x = \Delta y$) |
| `local_shape`, `global_shape` | Tuplas de forma |

La malla se genera automáticamente desde el archivo NetCDF de entrada, dividiéndose entre procesos MPI.

### 6.2 Condiciones de Contorno — [domain/boundary_registry.py](src/PyExner/domain/boundary_registry.py)

**`BoundaryManager`:** Clase que gestiona todas las condiciones de contorno.

- Lee las definiciones de contorno del YAML.
- Usa **polígonos** para identificar celdas de contorno (JIT-compilado con `point_in_polygon`).
- Calcula índices de celdas interiores adyacentes para condiciones reflectivas/transmisivas.
- `apply(state, time)` aplica todas las condiciones de contorno secuencialmente (JIT-compilado).

### 6.3 Tipos de Condición de Contorno

**Directorio:** [src/PyExner/domain/boundaries/](src/PyExner/domain/boundaries/)

| Clase | Registro | Descripción |
|:--------------------------------|:---------------------------|:------------------------------------------------|
| `Roe_`<br>`Reflective Boundary`  | `"Roe Reflective"`         | Pared sólida: escalares copiados, componente normal del momento reflejada |
| `Roe_`<br>`Transmissive  Boundary`| `"Roe Transmissive"`       | Salida libre: todos los campos copiados desde la celda interior |
| `RoeExner_`<br>`Reflective Boundary`| `"Roe Exner Reflective"`| Pared sólida con campos Exner ($z_b$, $G$, $n$) |
| `RoeExner_ TransmissiveBoundary`| `"Roe Exner Transmissive"`| Salida libre con campos Exner         |
| `RoeExner_`<br>`Zero MomentumBoundary`| `"Roe Exner ZeroMomentum"`| Fija $hu = hv = 0$ en la frontera (zona de remanso) |

### Definición en YAML:

```yaml
boundaries:
  wall_left:
    type: Reflective           # Tipo de condición
    polygon:                    # Vértices del polígono que define la frontera
      - [0.0, 0.0]
      - [0.01, 0.0]
      - [0.01, 10.0]
      - [0.0, 10.0]
    values: [NaN, 0.0, NaN, NaN]  # Valores forzados (NaN = no forzar)
    normal: [-1.0, 0.0]           # Normal exterior unitaria
```

El `flux_scheme` se concatena con el `type` para buscar en el registro (ej: `"Roe Exner"` + `"Reflective"` → `"Roe Exner Reflective"`).

Todas las clases se registran como **pytree de JAX** para ser compatibles con `jax.jit`.

---

## 7. Módulo `parallel/` — Computación Distribuida MPI

**Archivo:** [src/PyExner/parallel/mpi_utils.py](src/PyExner/parallel/mpi_utils.py)

### Clase `Parallel`

Encapsula la configuración MPI:

| Atributo | Descripción |
|---|---|
| `parNx`, `parNy` | Número de particiones en X e Y |
| `cart_comm` | Comunicador cartesiano 2D |
| `rank`, `size` | Rango y número total de procesos |
| `coords` | Coordenadas del proceso en la topología |
| `dims` | Dimensiones de la topología `(parNy, parNx)` |
| `neighbors` | Diccionario con rangos vecinos: `north`, `south`, `east`, `west` |

### Intercambio de Halos (`make_halo_exchange`)

Definido en cada kernel de solver ([kernels/roe.py](src/PyExner/solvers/kernels/roe.py), [kernels/roe_exner.py](src/PyExner/solvers/kernels/roe_exner.py)).

- Implementa un intercambio circular de celdas fantasma (ghost cells) de 1 celda de ancho.
- Usa `mpi4jax` para comunicación MPI compatible con JAX (`sendrecv` JIT-compilado).
- Orden de envío: west → north → east → south.
- Orden de recepción: east → south → west → north.

---

## 8. Módulo `io/` — Entrada/Salida

**Directorio:** [src/PyExner/io/](src/PyExner/io/)

### `PnetCDFStateIO` — [io/pnetcdf_reader.py](src/PyExner/io/pnetcdf_reader.py)

Maneja toda la lectura/escritura de datos en formato **PnetCDF** (NetCDF paralelo).

| Método | Descripción |
|---|---|
| `generate_mesh()` | Lee coordenadas globales, divide el dominio entre procesos MPI, crea y retorna un `Mesh2D` |
| `read_state(state, mesh, config)` | Lee variables del NetCDF de entrada → llena el estado (con padding de halos si MPI). También inicializa el archivo de salida |
| `write_state(state, mesh, mask)` | Escribe el estado actual al NetCDF de salida (escritura paralela colectiva) |

Soporta lectura de parámetros de sedimentos desde la configuración YAML (fracción, diámetro $d_{50}$, densidad, flujos de erosión/deposición, porosidad bulk).

### Archivos vacíos (reservados):
- [io/diagnostics.py](src/PyExner/io/diagnostics.py) — Para diagnósticos futuros.
- [io/visualizer.py](src/PyExner/io/visualizer.py) — Para visualización futura.

---

## 9. Módulo `utils/` — Constantes y Utilidades

**Directorio:** [src/PyExner/utils/](src/PyExner/utils/)

### Constantes — [utils/constants.py](src/PyExner/utils/constants.py)

| Constante | Valor | Descripción |
|---|---|---|
| `g` | `9.81` | Aceleración de la gravedad |
| `DRY_TOL` | `1e-3` | Tolerancia para celdas secas |
| `SED_TOL` | `1e-4` | Tolerancia para sedimento mínimo |
| `VEL_TOL` | `1e-6` | Tolerancia para velocidades pequeñas |
| `TIMESTEP_TOL` | `5e-7` | Tolerancia para el integrador temporal |
| `FLUX_TOL` | `1e-12` | Tolerancia para flujos |

---

## 10. Tests y Casos de Ejemplo

**Directorio:** [tests/](tests/)

### Casos disponibles:

| Caso | Directorio | Esquema | Descripción |
|---|---|---|---|
| **1D Dam Break** | `tests/runtime/1D_dambreak/` | Roe | Rotura de presa 1D clásica |
| **2D Dam Break** | `tests/runtime/2D_dambreak/` | Roe | Rotura de presa 2D simétrica |
| **Symmetrical Dam Break** | `tests/runtime/symmetrical_dambreak/` | Roe | Dam break simétrico |
| **Erodible Channel** | `tests/runtime/erodible_channel/` | Roe Exner | Canal erodible con transporte de sedimentos |
| **L-Domain** | `tests/runtime/L_domain/` | Roe Exner | Dominio en forma de L con erosión |

Cada caso incluye: script de construcción de input, archivo YAML de configuración, script de ejecución (`run.py`), y script de análisis (`analyze_data.py`).

---

## 11. Diagrama de Flujo de una Simulación

```
┌─────────────────────────────────────────────────────┐
│                    run_driver()                      │
├─────────────────────────────────────────────────────┤
│  1. Leer YAML config                                │
│  2. Parallel(params)  ← Topología MPI cartesiana    │
│  3. PnetCDFStateIO    ← Abrir archivos NC           │
│  4. generate_mesh()   ← Dividir dominio             │
│  5. create_empty_state() ← RoeState o RoeExnerState │
│  6. read_state()      ← Leer condiciones iniciales  │
│  7. BoundaryManager() ← Configurar contornos        │
│  8. create_solver_bundle() ← Roe o Roe Exner        │
│  9. create_integrator_bundle() ← FE o SSPRK2        │
│                                                      │
│  10. integrator.run_fn()                             │
│      ┌──────────────────────────────────────┐        │
│      │  while time < end_time:              │        │
│      │    jax.lax.while_loop:               │        │
│      │      ├─ compute_dt (CFL)             │        │
│      │      ├─ solver.step_fn()             │        │
│      │      │   ├─ Roe solve 2D (SWE)      │        │
│      │      │   ├─ [Momentum corrections]   │        │
│      │      │   ├─ Boundary apply           │        │
│      │      │   ├─ Halo exchange (MPI)      │        │
│      │      │   ├─ [compute_G + Exner]      │        │
│      │      │   └─ [Halo z_b + n_b]         │        │
│      │      └─ time += dt                   │        │
│      │    io.write_state()  ← salida NC     │        │
│      └──────────────────────────────────────┘        │
│                                                      │
│  return state, (X, Y)                                │
└─────────────────────────────────────────────────────┘
```

Los pasos entre `[corchetes]` solo aplican al solver `"Roe Exner"`.

---

## 12. Sistema de Registros (Registry Pattern)

PyExner usa un **patrón de registro con decoradores** para permitir extensibilidad. Los tres registros son:

| Registro | Decorador | Uso |
|---|---|---|
| `STATE_REGISTRY` | `@register_state("nombre")` | Estados (variables conservativas) |
| `SOLVER_REGISTRY` | `@register_solver_bundle("nombre")` | Solvers de Riemann |
| `INTEGRATOR_REGISTRY` | `@register_integrator_bundle("nombre")` | Integradores temporales |
| `BOUNDARY_REGISTRY` | `@register_boundary("nombre")` | Condiciones de contorno |

Para añadir un nuevo solver, estado o integrador, basta crear una nueva clase/función con el decorador correspondiente e importarla en el `__init__.py` del módulo.
