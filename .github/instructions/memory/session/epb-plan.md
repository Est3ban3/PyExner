# Plan de extension EPB_TwoFluid en PyExner

Fuente: Documentacion/epb_extension_plan.tex + ARCHITECTURE.md + documentacion.tex

## Objetivo
Anadir rama fisica `EPB_TwoFluid` (modelo 2.5D de dos fluidos para Burbujas de Plasma
Ecuatoriales) reutilizando infraestructura (driver, mesh, MPI, contornos, integradores, I/O).
NO modificar comportamiento de `Roe` ni `Roe Exner`.

## Decisiones acordadas
- Cierre de presion: ISOTERMO, p_alpha = n_alpha k_B T_alpha (T dado).
- Geometria: 2.5D plano (x,z); 3 componentes de corriente, solo x,z discretizadas.
- Estrategia: CONSERVADORA, cambios minimos.
- Mantener bitacora: Documentacion/EPB_CHANGELOG.md.

## Sistema continuo
dQ/dt + dF/dx + dG/dz = S(Q)
Q = [n_i, n_e, j_ix, j_iy, j_iz, j_ex, j_ey, j_ez]^T  (8 componentes, ORDEN CRITICO)
3 piezas SEPARADAS: (1) transporte hiperbolico conservativo, (2) fuentes rigidas
(colisiones+electrodinamica), (3) solve eliptico E = E0 - grad(phi).

## Hechos del framework (verificados en codigo)
- Registros con decorador: STATE_REGISTRY, SOLVER_REGISTRY, INTEGRATOR_REGISTRY, BOUNDARY_REGISTRY.
- Nombre boundary = flux_scheme + " " + type (ej. "EPB_TwoFluid Transmissive").
- BaseState (state/base.py) hardcodea h, hu, hv; metodos usan fields(cls) -> ya genericos.
- SimState (integrators/registry.py) anota state: BaseState -> revisar para estado arbitrario.
- SolverBundle: name, config, mask_fn, init_fn, step_fn, compute_dt_fn.
- SolverConfig: mpi_handler, boundaries, dx, halo_exchange, compute_G, compute_n.
- halo_exchange se define por kernel (mpi4jax sendrecv JIT).

## Fases (orden de ejecucion)
1. Generalizacion minima framework: utils de estado sobre dataclass arbitrario; tipos del
   integrador aceptan pytree arbitrario; back-compat Roe/Roe Exner. Riesgo: sobre-generalizar.
2. Alta EPB_TwoFluid: nuevo estado conservado (8 campos) + register_state("EPB_TwoFluid");
   solver bundle + register_solver_bundle("EPB_TwoFluid"); activar imports en __init__.
3. Slice hiperbolico puro: kernels F(Q), G(Q) jax.numpy (cuidado signos flujo electronico,
   F6 = -j_ex^2/(e n_e) - (e/M_e) p_e); cotas de onda HLL/HLLC; dt por onda max + MPI.MIN.
   NO reutilizar eigensistema SWE.
4. Contornos + I/O: boundary transmisivo 8 campos "EPB_TwoFluid Transmissive"; lectura/
   escritura NetCDF 8 variables (I/O ya itera campos dinamicamente).
5. Fuentes rigidas S(Q): gravedad, E, B, colisiones in/en/ei; kernel separado; solve eliptico
   phi -> E = E0 - grad(phi); parametros fondo g, E_xyz, B0, U_xyz, nu_in, nu_en, nu_ei.
6. Integrador IMEX: explicito transporte / implicito fuentes locales por celda; mismo contrato
   run_fn del framework.
7. Validacion: wiring, orden de Q, signo flujo electronico, contornos, estado uniforme, halos 8 campos.

## Estado actual
Fase 1 COMPLETADA (solo tipos: alias State=Any en state/base.py; SimState.state y
run_fn_forwardeuler usan State). BaseState utils ya genericos (fields(cls)). EPB state NO
necesita heredar BaseState.
Fase 2 COMPLETADA. Archivos nuevos:
- state/epb_twofluid_state.py: EPBTwoFluidState (8 campos), EPB_FIELD_ORDER, register_state+pytree.
- solvers/kernels/epb_twofluid.py: make_halo_exchange, halo_exchange_all (operativos);
  transport_step (PLACEHOLDER identidad) y compute_dt_2D (PLACEHOLDER dx) -> reemplazar en Fase 3.
- solvers/epb_twofluid_solver.py: SolverBundle "EPB_TwoFluid" (config/mask/init/step/compute_dt).
- inits de state y solvers actualizados.
Registries ahora: ['Roe','Roe Exner','EPB_TwoFluid']. Wiring + estado uniforme validados.
mask_fn devuelve stack([blocked,b_mask]) (mask[0] compat integrador).
Hallazgos: SSPRK2.py obsoleto/no registrado (codigo muerto); test_mesh.py usa API vieja
de Mesh2D; entorno WSL SI tiene CUDA (JAX en GPU OK, falta solo nvcc que no hace falta).
Crash al importar mpi4py = MPICH GPU-aware falla gpu_mem_hook_init en WSL.
WORKAROUND pruebas locales: export MPIR_CVAR_ENABLE_GPU=0 (con eso todo PyExner importa).
En cluster real usar MPI4JAX_USE_CUDA_MPI=1 en su lugar.

Fase 3 COMPLETADA. kernels/epb_twofluid.py ahora tiene fisica HLL real:
- EPBPhysParams(NamedTuple e,kB,Mi,Me,Ti,Te=1.0; ci/ce con math.sqrt; from_params lee params['epb']).
- stack_state/unstack_state, _velocities (signo carga: vi=+ji/(e ni), ve=-je/(e ne)).
- physical_flux(arr,phys,axis 'x'/'z'); _wave_bounds (Davis por bloque, ion idx {0,2,3,4}, ele {1,5,6,7});
  hll_flux; transport_step(state,dt,dx,mask,phys) barridos x/z; compute_dt_2D jit static_argnums=(3,).
- Convencion signos: izq +=Fhat, der -=Fhat, Q_new=Q-(dt/dx)*div (igual que Roe).
SolverConfig gana phys:object=None (registry.py). solver epb: config build phys, step/dt pasan config.phys.
CLAVE jit: float(jnp.sqrt(...)) falla en jit aunque phys sea static -> usar math.sqrt.
Validado: A interior uniforme div=0; B F6=-2.125 ok; C F3=10/3 ok; D HLL consistente;
E positividad 200 pasos min n=0.712>0, dt finito; F registries intactos.
Proxima: Fase 4 (contornos EPB_TwoFluid Transmissive + I/O NetCDF 8 campos).

Fase 4 COMPLETADA.
- domain/boundaries/epb_transmissive.py: EPB_TransmissiveBoundary registrado como
  "EPB_TwoFluid Transmissive". apply = zero-gradient (copia 8 campos interior->borde,
  SIN reflejar momento, distinto del caso SWE). pytree. Recibe mask/normal/boundary_indices/
  interior_indices (rama "Transmissive" de BoundaryManager).
- state/epb_twofluid_state.py: anadido to_host()=tree_map(device_get,self) (lo usa write_state).
- domain/boundaries/__init__.py y domain/__init__.py exponen EPB_TransmissiveBoundary (activa registro).
- I/O SIN cambios: read_state/write_state ya iteran fields(state) -> manejan 8 campos. Input NetCDF
  debe traer las 8 variables (n_i,n_e,j_ix,j_iy,j_iz,j_ex,j_ey,j_ez). diagnostics.py vacio.
Validado: A boundary registrado; B copia zero-gradient ok; C interior intacto; D to_host->numpy.
Proxima: Fase 5 (fuentes rigidas S(Q) + solve eliptico phi).

Fase 5 COMPLETADA. kernels/epb_sources.py (SEPARADO del transporte):
- EPBSourceParams(NamedTuple): gx/gy/gz,E0x/y/z,Bx/y/z,Ux/y/z,nu_in,nu_en,nu_ei,poisson_iters.
  Defaults NULOS (sin fuentes=transporte puro intacto). props g,E0,B,U. from_params lee params['epb']['sources'].
- source_term(arr,phys,src,E)->(...,8) filas continuidad=0. j son CORRIENTES (ji=e ni vi, je=-e ne ve).
  S_ji=e ni g + e^2 ni/Mi E + e/Mi ji×B - nu_in(ji - e ni U).
  S_je=-e ne g + e^2 ne/Me E - e/Me je×B - nu_en(je + e ne U) - nu_ei(je + (ne/ni)ji).
  Signos: gravedad OPUESTA ion/ele; electrica MISMO signo (e^2 n/M); magnetica q/M j×B.
- solve_phi(arr,dx,n_iter) jit static(2): Poisson lap(phi)=div(J), J=ji+je plano (idx 2/5 x, 4/7 z),
  Jacobi fori_loop, gauge media cero, bordes periodicos (roll) -> infra, falta BC fisica/halo/CG.
- electric_field(phi,src,dx)=E0-grad phi (corrige Ex,Ez; Ey=E0y).
- apply_sources_explicit: Euler explicito SOLO test (orden phi->E->S->update). NO en step_fn.
SolverConfig gana src:object=None. config_fn_epb construye src. step_fn SIGUE transporte puro (Fase 6 integra).
Validado: A S=0 sin params; B grav gz=-1 ion -2/ele +2; C E electrica mismo signo +2; D j×B=(0.1,0,0.3);
E nu_in drag -0.6; F Poisson residual 4e-16; G E=E0 si phi=0; H densidades intactas.
Proxima: Fase 6 (integrador IMEX: transporte explicito + fuentes implicitas).

Fase 6 COMPLETADA.
- kernels/epb_sources.py: _cross_matrix(Bx,By,Bz) K j=j×B; implicit_source_solve(state,dt,phys,src,dx):
  fuente AFIN S(j)=A j + b (densidades no se sourcean). Resuelve (I-dt A) j=j+dt b por celda 6x6 con
  jnp.linalg.solve vectorizado. A_ii=(e/Mi)K-nu_in I; A_ee=-(e/Me)K-(nu_en+nu_ei)I; A_ei=-nu_ei (ne/ni)I
  (bloque-triangular inf, ion no depende de ele). b_i=e ni g+e^2 ni/Mi E+nu_in e ni U; b_e analogo.
  E LAGGED del solve_phi (rigidez local implicita, acoplamiento electrostatico explicito).
- SolverBundle gana source_fn:Callable=None (hidraulicos lo dejan None). registry.py.
- epb_twofluid_solver.py: source_fn_epb jit static(3) = implicit_source_solve+halo+BC. Anadido al bundle.
- integrators/imex.py: integrador "IMEX" espejo de Forward Euler con _imex_step=step_fn(transporte)+
  source_fn(fuentes). Si no hay source_fn -> = Forward Euler. integrators/__init__.py importa.
Splitting de Lie 1er orden con dt convectivo.
Validado: A IMEX registrado ['Forward Euler','IMEX']; B densidades intactas; C imp~exp dt=1e-4 diff 1e-8;
D rigidez nu=1e6 implicito 3e-7 acotado vs explicito -3e5 explota; E giro magnetico |j| 0.4995~0.5;
F SOLVER_REGISTRY intacto.
Proxima: Fase 7 (validacion incremental end-to-end).





Fase 7 COMPLETADA. PLAN EPB_TwoFluid COMPLETO (Fases 1-7).
- tests/solvers/test_epb_twofluid.py: bateria autoejecutable (pytest + runner __main__), 7 pruebas
  mapeadas a los 6 items del plan. Dominio sintetico 1 rango parNx=parNy=1, malla 8x8 dh=1,
  4 contornos transmisivos EPB reales via BoundaryManager.
- BUGFIX epb_twofluid_solver.py: source_fn_epb tenia static_argnums=(3,) (marcaba mask); el no-array
  estatico es config (idx 4, contiene Parallel) -> corregido a (4,) como step_fn_epb. Sin esto JAX
  intentaba trazar Parallel como abstract array y abortaba.
Pruebas: 1 wiring (3 registros + no-regresion Roe); 2 orden stack/unstack roundtrip 8 hojas;
3 signos flujo (F_ne=-jex/e, F_ni=+jix/e, ion presion +, electron presion -); 4 contornos nombre/
BoundaryManager 4 handlers; 5 estado uniforme: transporte preserva interior exacto atol=1e-12 +
fuentes nulas preservan densidad; 6 halo 8 campos sin mezcla; 7 conservacion+positividad 20 pasos IMEX.
Resultado: 7/7 passed (x64, MPIR_CVAR_ENABLE_GPU=0).
HALLAZGOS CLAVE:
- Invariante uniforme se verifica sobre TRANSPORTE PURO ("en ausencia de fuentes" del plan), NO sobre
  IMEX completo: el solve global de phi responde a gradientes espurios de borde en dominio sintetico
  sin ghost layer. Conservacion bajo IMEX validada por separado.
- Poligonos de contorno deben dimensionarse al paso de malla para capturar el anillo de CENTROS
  ((i+0.5)*dh); bandas mas estrechas que dh no capturan celdas -> contornos inactivos (error silencioso).
Bitacora: Documentacion/EPB_CHANGELOG.md seccion "Fase 7" con modelos LaTeX, archivos, validacion,
riesgos y cierre del plan.

## Validacion fisica (escalera de 4 niveles) — COMPLETA
Entregables: SCRIPTS en tests/runtime/EPB/ (preferencia del usuario). Bitacora en
Documentacion/EPB_CHANGELOG.md secciones "Validacion fisica — Nivel N".
- Nivel 1 (smoke_epb.py): driver end-to-end. PASSED. Genera input.nc + input_epb.yaml, run_driver,
  valida via pnetcdf (NO xarray: scipy no lee CDF-5/NC_64BIT_DATA). +14% deriva de masa = artefacto
  de contorno transmisivo (no del esquema).
- Nivel 2 (advection_epb.py): adveccion analitica plasma frio (Ti=Te=0), periodico via roll. 3/3.
  ion bump->+v0, ele bump->-v0; convergencia L1 orden ~0.73; masa conservada 1e-13 (confirma que la
  deriva del Nivel 1 era de contorno).
- Nivel 3 (sources_epb.py): sub-bloques de fuentes aislados sobre estados uniformes. 3/3.
  A giro magnetico (angulo->1rad, A-estable); B relajacion colisional (backward Euler exacto j0/(1+nu dt)^N);
  C Poisson (phi vs analitico 8.7e-4). HALLAZGO: desacoplamiento par/impar -- solve_phi usa laplaciano
  5pt pero _central_diff es paso-2 -> div(grad) compuesto deja modo checkerboard ~3e-2. Medir con
  operador consistente. Riesgo para produccion: grad/div consistentes con el laplaciano (o malla
  escalonada / proyeccion CG).
- Nivel 4 (rt_instability_epb.py): inestabilidad de intercambio (RT generalizada), plasma frio,
  B=B y, g=-g z. Harness propio (transporte periodico + implicit_source_solve). PASSED.
  CLAVE: dominio DOBLEMENTE PERIODICO (solve_phi usa BC periodicas via roll). Paredes transmisivas
  fugan masa 39%; paredes reflectantes conservan masa pero CHOCAN con BC periodicas de solve_phi
  (config estable explota ~1e4, n<0). Contraste correcto = ESTRATIFICADO (banda densa, bottomside
  RT-inestable) vs UNIFORME (control sin energia libre), misma siembra+masa media, medida
  A=Σ(n-<n>_x)^2. Resultado: estratificado crece 5.75x (pico ~12), uniforme DECAE 0.43x, masa 0%,
  positividad ok. gamma medido 0.347 vs sqrt(g/Ln)=1.826 (ratio 0.19, WARN no bloquea: difusion upwind
  HLL + contaminacion par/impar + modo k finito). El SIGNO del contraste es lo robusto.
  NOTA: descartado contraste top-heavy/stable con medida A -> la cizalla de deriva gravitacional
  diferencial (iones +x, ele -x) genera estructura-x en ambas orientaciones y enmascara la firma RT.

ESTADO FINAL: EPB_TwoFluid implementada (Fases 1-7) + validada fisicamente (Niveles 1-4). COMPLETO.
Trabajo de produccion pendiente (no solicitado): BC no-periodicas en solve_phi (Neumann/Dirichlet
consistentes con transporte), grad/div consistentes con laplaciano 5pt (anti checkerboard),
validacion multi-rango en cluster.

## Camino critico hacia EPB CUANTITATIVA — Pasos 1-2 COMPLETOS
Modulo NUEVO: src/PyExner/solvers/kernels/epb_ionosphere.py (SEPARADO; cuarta pieza: quimica).
- SI/SI_CONST + M_OPLUS (16 amu). FRegionParams: capa F ELEVADA post-atardecer/PRE
  (h_peak=400km, n_max=1e12, H_chapman=50km, beta0=2e-4@300km H_beta=30km, nu0=0.5@300km H_nu=40km,
  Ti=Te=1000K, B_surf_eq=3.12e-5, g_surf=9.8). OJO: valores iniciales (h_peak=350,beta0=1e-3,nu0=1)
  daban beta dominante en TODO el dominio -> sin region inestable; se subio la capa y se bajaron
  beta/nu a valores de bottomside elevado (condicion real de EPB).
- Perfiles: chapman_layer (n0 Chapman-alpha), exp_profile (nu_in,beta), dipole_B (B_eq(RE/(RE+h))^3),
  gravity (g0(RE/(RE+h))^2). build_background -> EPBBackground(z,n0,nu_in,beta,prod,B,g) con P=beta n0
  AUTO-CONSISTENTE (n0 equilibrio exacto). physparams_SI -> EPBPhysParams SI (ci~721, ce~1.23e5 m/s).
- QUIMICA: chemistry_implicit_step(n_i,n_e,dt,prod,beta): backward Euler n=(n+dt P)/(1+dt beta),
  A-estable e INCONDICIONALMENTE POSITIVA (apta para beta rigida). chemistry_step_state via .replace().
  CLAVE arquitectura: prod/beta son ARRAYS broadcastables (NO van en EPBSourceParams porque src viaja
  en config ESTATICO del jit -> arrays no caben). Separacion: transporte|momento|electrostatica|quimica.
- Diagnosticos: density_scale_length (L_n=n/|dn/dz|), collisional_growth_rate
  gamma = g/(nu_in L_n) - beta (RT COLISIONAL Sultan, NO el inercial sqrt(g/L_n) del Nivel 4).
Script: tests/runtime/EPB/ionosphere_epb.py (4 bloques, TODOS PASS, x64).
A unidades SI en rango; B equilibrio P=beta n0 punto fijo 1e-14; C operador quimico exacto+positivo+
rigido; D RESULTADO CONCRETO: umbral altura EPB ~252km, bottomside inestable 275-350km tau=15-28min,
e-folding tipico ~19min = RANGO OBSERVADO de aparicion EPB post-atardecer. gamma max a 325km (15.6min).
Por debajo del umbral recombinacion estabiliza; max gamma en bottomside elevado (ni pico L_n->inf ni
abajo nu_in/beta grandes). Bitacora: EPB_CHANGELOG.md seccion "Camino critico ... Pasos 1-2".
PROXIMOS pasos camino critico (no hechos): 3 transporte 2o orden MUSCL/WENO+limitador positivo;
4 solve_phi produccion (consistente anti-checkerboard + BC Neumann + multigrid/CG); 5 conductividad
capa E Sigma_E; 6 flux-tube integrada dipolar; 7 disparador PRE E0(t)+validacion observacional.
PENDIENTE wiring: cablear quimica al source_fn con nu_in(h),beta(h) como campos (hoy validada aparte).

## Camino critico — Paso 3: transporte 2o orden MUSCL COMPLETO
epb_twofluid.py (anadido SIN tocar 1er orden, Fase 7 sigue 7/7):
- _minmod3, _mc_slope (limitador MC monotonized central, TVD).
- _muscl_faces_periodic, muscl_hll_flux_periodic (reconstruccion lineal caras + HLL, periodico via roll).
- _muscl_divergence, transport_step_muscl (paso completo).
CLAVE: MUSCL 2o-orden-espacial REQUIERE SSP-RK2 en tiempo. 1er intento con Euler explicito EXPLOTO
(RT n~-1e120, inestabilidad lineal del 2o orden espacial centrado + Euler). SSP-RK2 Heun:
Q1=Q+dt L(Q); Qn+1=0.5Q+0.5(Q1+dt L(Q1)), L=-divF/dx. Combinacion convexa de pasos Euler TVD ->
estable + positivo CFL<=1. TVD => positividad sin floor (caras acotadas entre vecinos).
Script tests/runtime/EPB/muscl_epb.py 4 bloques TODOS PASS:
A orden L1 adveccion suave: 1er=0.72 MUSCL=1.66 (MC recorta extremo gaussiano -> 1.66 no 2.0, esperado).
B frente afilado pluma: error L1 MUSCL 0.27x del 1er orden, rampa 18 vs 64 celdas. OJO medida: variacion
total NO sirve (se conserva bajo difusion monotona) -> usar error L1 vs exacto y ancho de rampa.
C positividad deplec 99% 400 pasos: min n>0 sin floor.
D RT doble-periodico (params Nivel 4): gamma 1er=0.105(ratio0.06) MUSCL=0.531(ratio0.29) =5x;
amplitud x947 vs x9.6. HALLAZGO: gamma subestimado del Nivel 4 era SOBRE TODO difusion 1er orden.
Residual 0.29x confirma causa restante = par/impar de solve_phi (Paso 4).
NOTA: transport_step_muscl es PERIODICO; step_fn por defecto SIGUE 1er orden (no romper ramas).
Contornos fisicos MUSCL necesitan ghost cells (produccion). Coste x2 por paso (2 flujos RK2).
Bitacora EPB_CHANGELOG.md "Paso 3: transporte de 2o orden (MUSCL)".

## Camino critico — Paso 4: solve electrostatico CONSISTENTE COMPLETO
epb_sources.py seccion electrostatica reescrita (solve_phi/electric_field MISMA interfaz):
- _grad_forward G+f|i=(f_{i+1}-f_i)/dx ; _div_backward D-u|i=(u_i-u_{i-1})/dx ; D-G+ = L_5pt EXACTO.
- _poisson_divergence (RHS backward + gauge media cero) ; _poisson_cg (CG SPD, A=-L).
- _central_diff conservada solo diagnostico (OBSOLETA). electric_field usa grad FORWARD (consistente).
HALLAZGO Nivel 3 resuelto: viejo resolvia L_5pt phi = div_central J pero E=E0-grad_central phi ->
div_c(grad_c)=laplaciano CENTRAL ancho (2dx) != L_5pt -> modo checkerboard residual O(1e-2).
Fix: tripleta D-/G+/L_5pt => D-(J-G+phi)=D-J-Lphi=residual CG ~ maquina (limpieza EXACTA sin checkerboard).
Coste: campo O(dx) (medio celda desfase backward) a cambio de div exacta (trade MAC estandar).
CG BUG CRITICO: iterar de mas TRAS converger divide por ~0 y EXPLOTA (probado: 600 iters -> phi~1e3).
FIX: congelar iteracion al converger (active = rs > 1e-24*rs0 & denom>0 ; alpha=beta=0 si no activo).
Con la congelacion sobre-especificar n_iter es inocuo (default poisson_iters=200 OK).
Script tests/runtime/EPB/poisson_epb.py 4 bloques TODOS PASS:
A identidad D-G+ == L_5pt: 2.3e-13. B limpieza consistente 3.8e-12 vs central viejo 9.8e-2 (=checkerboard).
C CG converge <5 iters 6.5e-12 vs Jacobi(40)=8.5 (1.3e12x); manufacturada 3.7% (O(dx) esperado).
D control checkerboard: |L_5pt cb|=8/dx^2 (ACOPLA) vs |L_central cb|=0 (NUCLEO=modo libre, la patologia).
sources_epb.py (Fase 5) bloque C actualizado a tripleta consistente: C1=4.9% (O(dx), umbral 8e-2),
C2/C3 ~7e-12 maquina, C3b verifica checkerboard RESUELTO. Sin regresion: Fase 7 7/7, Fase 5 3/3.
IMPACTO RT Nivel 4 (reejecutado, solo nuevo solve): gamma 0.347->0.426, contraste 5.75x->10.4x,
pico x12->x21, masa/positividad intactas. PENDIENTE produccion: BC Neumann/Dirichlet, CG en MPI
(halo_exchange por matvec), precondicionador multigrid. Bitacora "Paso 4: solve electrostatico consistente".

## Camino critico — Paso 5: capa E como carga de conductancia COMPLETO
epb_sources.py seccion nueva (Paso 4 intacto): _face_avg, _div_sigma_grad (L_sigma=D-(sigma_f G+)),
_poisson_cg_sigma, solve_phi_sigma(arr,sigma,dx,n_iter) con SIGMA_FLOOR=1e-12 (SPD si n->0).
implicit_source_solve(...,sigma=None): con sigma usa solve_phi_sigma; None conserva camino Paso 4.
epb_ionosphere.py: shielding_factor F_s=sigma_F/(sigma_F+sigma_E); collisional_growth_rate_shielded.
Modelo: div(Sigma grad phi)=div(J_drive), Sigma=Sigma_F(x)+Sigma_E; gamma=F_s g/(nu_in L_n)-beta.
Script tests/runtime/EPB/elayer_epb.py 4 bloques TODOS PASS:
A equivalencia sigma=1 vs Paso4: 1.1e-14. B shunt uniforme sigma=1+R exacto phi/(1+R): maquina (R=1,3,9).
C limpieza sigma(x,z): residual 3.5e-10. D RT con shunt R=3 F_s=0.25: gamma 0.230->-0.172 (CRUZA umbral),
amplitud suprimida 177x = mecanismo dia/noche. RIESGO: media aritmetica de caras (armonica si burbuja
profunda n->0); residual 1e-10 hereda criterio congelacion CG. Bitacora "Paso 5: capa E ...".

## Camino critico — Paso 6: geometria flux-tube dipolar COMPLETO
epb_fluxtube.py (NUEVO, HOST-SIDE NumPy, precomputo que alimenta kernels JAX; perfiles reusados de
epb_ionosphere = unica fuente de verdad): ELayerParams (+ELAYER_NIGHT/DAY, capa E Chapman h_pE=105km
H_E=6km n_maxE 5e9 noche/1.5e11 dia); nu_in_total (DOS exp: termosfera F escala 40km + base E escala 7km,
nu(300)~0.5 nu(105)~3e3; corrige extrapolacion mono-exp Paso 2); pedersen_conductivity
sigma_P=ne/B nu Omega/(nu^2+Omega^2); dipole_line (r=L RE cos^2 lat, B dipolar, trunc h_min=90km);
flux_tube_quantities (Sigma_P^{F,E}=2 int sigma ds, N_FT, nu_eff, F_s); bottomside_Ln (Chapman analitico),
gamma_flux_tube, gamma_bottomside_max. Modelo gamma_FT(h_apex)=F_s(tubo) g/(nu_eff L_n)-beta (Sultan, sin viento/V_P).
Script tests/runtime/EPB/fluxtube_epb.py 4 bloques TODOS PASS:
A geometria (apex 1e-10, B dipolar 2.7e-16, pie OK); B convergencia trapecio nlat 401->1601 drift 4.1e-6;
C conductancias 350km Sigma_F 4.5S Sigma_E 0.63/18.8S F_s 0.88->0.19; D umbral.
RESULTADOS FISICOS: umbral nocturno h_apex~300km (sube vs 252km local Paso2: linea muestrea alturas mas
colisionales + remanente capa E, direccion correcta); e-folding min nocturno tau=25.8min @ h~340km
(gamma_max 6.4e-4 vs Sultan 5e-4..1.5e-3); topside estable; supresion diurna tau_min 169min (x6.5).
RIESGO: ion unico O+ tambien en E (alli NO+/O2+ m~30 factor 2 en sigma_E); sin vientos U_L^P; g,beta en apex.
Bitacora "Paso 6: geometria flux-tube integrada (dipolar)".

## Camino critico — Paso 7: disparador PRE y onset COMPLETO
epb_pre.py (NUEVO, HOST-SIDE NumPy, forzante lento escala horas -> E0x(t),R_E(t),perfiles a kernels):
PREParams ciclo diurno tipo Fejer V(t)=V_day cos(2pi(t-12)/24)+V_pre exp(-(t-t_pre)^2/2w^2)
(V_day=20,V_pre=30 m/s,t_pre=18.75LT,w=0.5h) + decaimiento logistico capa E en terminador (t_ss=18.2LT tau_E=0.4h);
vertical_drift, pre_electric_field (E0x=BV), elayer_density, layer_height (dh/dt=V trapecio piso 200km);
onset_prediction (perfil F desplazado rigido h_peak=h(t), capa E n_E(t), gamma=gamma_bottomside_max Paso6,
Gamma=int max(gamma,0)dt', onset al cruzar Gamma>=ln(1e3)~6.9 e-folds, semilla 1e-3 estandar Sultan/Huba).
Script tests/runtime/EPB/pre_onset_epb.py 4 bloques TODOS PASS:
A drift pico PRE 26.2 m/s @18.71LT (obs 20-60 @18-19:30); B capa F h_max 472km @19.57LT (obs 400-500);
C ONSET 19.93 LT (obs 19:30-22); D control SIN PRE Gamma a 02LT=0.36 e-folds (vs 15.6 con PRE) -> SIN onset.
EL CONTRASTE D ES EL RESULTADO CENTRAL: PRE es el disparador. Serie: gamma<0 hasta ~17:30, gamma_max 2.1e-3
(tau 8min) @20LT capa 465km F_s->1, madrugada gamma<0 (ventana finita). RIESGO: drift empirico suave (no
Scherliess-Fejer), subida rigida del perfil, criterio 6.9 e-folds supone semilla 0.1%. Morfologia no lineal SI
IMPRACTICABLE con inercia electronica explicita (CFL electron c_e~1.2e5 ~1e6 pasos/hora) -> requiere cierre
sin inercia (drift-difusion/vorticidad-potencial), siguiente etapa FUERA del camino critico. Bitacora "Paso 7".

## Adendo — Pluma SI 2D no lineal (plume_si_epb.py, fig5)
Primera corrida SI fisica: dominio 0-400km zonal x 200-800km altitud (96x144 dx=4.17km), Chapman post-PRE
(h_mF2=450km), B=2.58e-5T, nu_in=0.05. figures/fig5_pluma_si.png: plumas penetran h_mF2 -> topside ~600km
en 47min, v_max~250 m/s, deplecion ~5e-2 fondo, ascenso apice ~90 m/s (rangos observados).
3 LECCIONES NUMERICAS (el adimensional NO extrapola a SI):
1. Acoplamiento electrostatico LAGGED inestable con sigma_P fisica: resolver div(sigma_P grad phi)=div(J_total)
   da phi^{n+1}~phi_eq-phi^n (lazo marginal -> explosivo x5/paso dt=6s). CIERRE drive-driven (Ossakow): RHS solo
   corriente motriz gravitacional J_g=(n Mi g/B)x_hat -> phi diagnostico de n, unico acople temporal = transporte
   (gamma_RT<<1/dt) estable. implicit_source_solve acepta E_ext para esto.
2. NO transportar inercia de corrientes en regimen de derivas: Omega_i dt~1e3, el solve implicito regenera j cada
   paso (j diagnostica); advectar momento 8 campos hace v_f=j_f/(en_f)->inf en la deplecion (explota t~350s). Se
   advecta SOLO continuidad dt n_s+div(n_s v_s)=0 con deriva por especie (MUSCL escalar, limitador MC).
3. Regularizaciones FISICAS: (i) piso conductividad sigma_min=sigma_P(0.25 n_max) (sustituto 2D de conductancia FT);
   (ii) difusion sub-grid D~0.15(g/nu_in)dx (RT colisional gamma indep de k, corta modos de rejilla, Peclet~500
   escala 50km, tipo Zalesak). Bitacora EPB_CHANGELOG.md "Adendo — Pluma SI 2D no lineal".

## Auditoria de consistencia (2026-06-18)
Verificado que changelog/codigo/tests COINCIDEN. Todos los kernels documentados existen
(epb_ionosphere/twofluid/sources/fluxtube/pre). implicit_source_solve(...,sigma=None,E_ext=None) confirmado.
Bateria reejecutada HOY: 11/11 scripts PASAN (advection,sources,ionosphere,muscl,poisson,elayer,fluxtube,
pre_onset,rt_instability,smoke + test_epb_twofluid 7/7); plume_si compila + fig5 existe. Resultados coinciden:
elayer 177x, fluxtube tau 25.8min, onset 19.93LT, RT gamma 0.426 / 10.4x. DISCREPANCIA (esperada, no bug):
muscl_epb bloque D da hoy 0.157->0.623 (x3.96) vs changelog 0.105->0.531 (x5) porque Paso 4 reescribio solve_phi
(campo E que usa esa prueba); bloques A/B (transporte puro) sin cambio. CHANGELOG reestructurado: tabla Plan de
fases corregida (Fases 3-7 Completada), anadido INDICE MAESTRO al inicio (Capa->kernel->script->estado->resultado
+ comando reproducir), listas "Estado camino critico" intermedias marcadas como snapshots historicos, nota MUSCL
en Paso 3. Pendientes produccion sin cambio (BC no-periodicas+CG MPI, media armonica sigma, cierre sin inercia
electronica, vientos U_L^P, cablear nu_in(h)/beta(h) a source_fn).
