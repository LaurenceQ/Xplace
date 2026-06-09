# Route Segment Gradient Implementation Status

Date: 2026-06-04

## Implemented

- Added isolated route_grad sidecar CUDA implementation in `cpp_to_py/gputimer/core/route_grad/`.
- `RouteGradNetPrimitiveReverse` now reconstructs net delay candidates for direct-net and gate-net-pair DMP paths.
- Added analytic load-crossing derivative `d(load crossing time) / d(Elmore)` via implicit root differentiation:
  `dT/dE = -V_E / V_T`.
- Final net-delay slope includes threshold-adjust slew contribution: raw delay slope plus signed beta times raw slew slope.
- `compute_dmp_route_segment_soft_timing_grad()` now computes per-pin-slot net delay-vs-Elmore slopes with a temporary sidecar kernel and uses them in active AT reverse before the existing RC-tree reverse.
- The normal DMP timing/power path is not modified; the new kernel is launched only by the debug route gradient API.

## Validation

- Built with `cmake --build . -j8` and installed with `make install`.
- Python import in conda `gnn` loads `/research/d7/ascstd/qkduan25/Xplace/cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so` and exposes route_grad APIs.
- visible/ariane smoke:
  - edges: `1336274`, nonzero edge grads: `275775`, max abs edge grad: `0.0860506238`
  - nodes: `1460175`, nonzero node grads: `451505`, max abs node grad: `0.0003988264`
  - known edge adjoints: `1014260=0.010575464`, `1014259=0.008379194`, `1014264=0.001789853`, `942935=0.000175401`
  - known node adjoint: `1105069=2.537e-05`
- FD validator now prefers centered perturbation and falls back to one-sided only near invalid negative R/C. Edge FD remains unstable for small eps because winner/branch switches and objective numerical noise dominate; `1014264` and `942935` are not reliable FD oracles.

## Remaining Gap

- Cell-delay/root-PI/load-cap adjoint is still missing, so node-cap FD remains much larger than adjoint for cell-delay-dominant cases, e.g. ariane node `1105069` FD about `0.00238` vs adjoint about `2.54e-05`.
- Slew adjoint is not wired because current timing does not preserve a slew-winner predecessor and gate delay wrt input slew adjoint is not yet implemented.
- `cuobjdump --dump-resource-usage` shows `routeGradNetElmoreSlopeKernel` uses 126 regs and 1176B stack, similar to existing DMP direct/gate kernels; it is debug-only. Temporary GPU memory is two `num_pins*NUM_ATTR` float arrays.

## 2026-06-04 Update: Active Gate Root-Load To RC Moments

Implemented additional route_grad chain pieces:

- Added active gate root-load sidecar slope arrays keyed by `pin_slot`, not `arc*8`, to keep memory small.
- For each active gate winner, locally recomputes DMP gate delay with finite differences on `(C1, C2, rpi)` and accumulates `bar_C1`, `bar_C2`, `bar_rpi` during active AT reverse.
- Replaced the old Elmore-only RC reverse with a unified host reverse that:
  1. recomputes DMP moments `M/N/P` per RC tree,
  2. applies Elmore adjoints to `bar_M`/`bar_R`,
  3. maps root `bar_C1/C2/rpi` to root moment adjoints,
  4. reverses the DMP moment recurrence to edge resistance and node cap gradients.

Validation on `visible/ariane` after build/install:

- edge nonzero grads: `345328`, max abs edge grad: `0.084615`
- node nonzero grads: `487864`, max abs node grad: `0.002667`
- node `1105069`: FD `0.002384151`, adjoint `0.002293638` (close; previous adjoint was only about `2.54e-05`)
- known edge adjoints after root-load chain: `1014260=0.00438973`, `1014259=0.00465309`, `1014264=0.00160904`, `942935=0.00017813`

Remaining gaps:

- The active gate root-load primitive currently uses local finite differences on `(C1,C2,rpi)`; this should be replaced by the analytic implicit-solve reverse from `22_dmp_pi_rc_derivative_chain.md` for the final derivation-quality implementation.
- Slew adjoints are still not fully wired. In particular, downstream cell delay wrt input slew should create `bar_pin_slew`, and net sink slew should feed that back to Elmore/root waveform. Current DMP forward does not preserve a separate slew winner, so first implementation may need winner reconstruction or an active-winner approximation.
- FD edge checks remain noisy even with centered global perturbation due to winner/branch changes; add fixed-winner local checks before treating edge FD as a strict oracle.

## 2026-06-04 Update: Slew Adjoint And Centered FD

Implemented additional debug route-gradient pieces:

- Extended the isolated `route_grad` sidecar gate primitive from delay-only root-load slopes to pin-slot keyed slopes for:
  - gate delay wrt `(C1, C2, rpi, input_slew)`,
  - gate output slew wrt `(C1, C2, rpi, input_slew)`.
- Added `bar_pin_slew` to the host reverse pass. Current first version uses the active AT winner as the slew reverse path approximation because DMP forward does not preserve a separate slew-winner predecessor.
- Wired downstream gate delay input-slew adjoints back to upstream `bar_pin_slew`, and wired gate output-slew adjoints back to root PI and upstream input slew.
- Wired net sink slew adjoints to the existing net slew-vs-Elmore slope sidecar. Source-waveform adjoint through the net primitive is still not fully analytic.
- Changed route-segment FD gradcheck to prefer centered finite difference and fall back to one-sided difference only when the negative perturbation would make R/C invalid.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- smoke: edge grads `1336274`, nonzero `345328`, max abs `0.084615`; node grads `1460175`, nonzero `487864`, max abs `0.002667`; edge-cap grads nonzero `451508`, max abs `0.002664`.
- known adjoints: `edge 1014260=0.00438973`, `edge 1014259=0.00465309`, `edge 1014264=0.00160904`, `edge 942935=0.000178126`, `node 1105069=0.00229364`.
- centered FD spot check at `eps_rel=1e-3`, node `eps_abs=1e-4`: `node 1105069` FD `0.00299473` vs adjoint `0.00229364`; edge samples remain noisy and can disagree in sign because route/timing winner branches switch under perturbation.

Remaining gaps:

- Active gate primitive slopes are still local finite differences around DMP's current branch. Replace with the analytic implicit-solve reverse from `22_dmp_pi_rc_derivative_chain.md` for final derivation-quality code.
- Net source-waveform reverse now has a local finite-difference sidecar for active net candidates, but it still needs replacement by analytic crossing/waveform reverse for derivation-quality code.
- DMP forward still lacks a saved slew winner. The current slew chain is an active-AT-winner approximation; full slew reverse needs either storing slew winner or reconstructing it explicitly.

## 2026-06-04 Update: Net Driver-Wave Reverse

Implemented the missing net load-crossing driver-wave chain in the isolated debug route-gradient sidecar:

- Extended `RouteGradNetPrimitiveReverse` with per sink pin-slot winner metadata:
  - `driver_root_slot`,
  - `driver_input_slew_slot`,
  - delay/slew slopes wrt `(C1, C2, rpi, input_slew)`.
- `writeSlopeForNetArc()` still selects the same active direct-net/gate-net-pair candidate as before, but now computes local slopes only for the selected candidate.
- Added local finite-difference primitive for net sink delay/load slew wrt the selected driver waveform root PI and input slew. This covers the mem formula path:
  `driver root PI -> driver waveform w -> load crossing L_alpha -> net delay/slew`.
- Reverse now accumulates net delay/slew adjoints into:
  - sink Elmore adjoint,
  - driver root `(C1, C2, rpi)` adjoints,
  - upstream `bar_pin_slew` for source-slew dependence.
- Normal DMP timing kernels remain unchanged; all arrays are debug route-gradient temporaries. Additional net sidecar memory is per pin-slot: 2 int arrays plus 10 float arrays.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- smoke after net driver-wave reverse: edge nonzero `524277`, max abs `0.0844484`; node nonzero `746594`, max abs `0.00379211`; edge-cap nonzero `692758`, max abs `0.00370573`.
- known adjoints: `edge 1014260=0.00546845`, `edge 1014259=0.00530568`, `edge 1014264=0.00164051`, `edge 942935=0.000178332`, `node 1105069=0.00274690`.
- centered FD spot check: `node 1105069` FD `0.00298802` vs adjoint `0.00274690`; edge FD remains noisy due winner/branch switching.

Remaining gaps after this update:

- Gate primitive and net driver-wave primitive are still local finite differences around DMP's current branch. Replace with analytic reverse from `22_dmp_pi_rc_derivative_chain.md`, especially `J_x^T mu = bar_x` and crossing reverse `bar_w -= bar_T V_w/V_T`.
- Current slew propagation still follows active AT winner unless the net/gate sidecar reconstructs the active candidate. Full slew correctness needs saved slew winner or explicit slew-winner reconstruction.
- FD validation is global rerun validation. Add fixed-winner local checks for edge samples before treating edge FD as a strict oracle.


## 2026-06-04 Update: Explicit Slew-Winner Reconstruction

Implemented the slew-winner split in the isolated debug route-gradient sidecar:

- Split net primitive winner metadata into independent delay and slew winners:
  - `delay_driver_root_slot`, `delay_driver_input_slew_slot`,
  - `slew_driver_root_slot`, `slew_driver_input_slew_slot`.
- `RouteGradNetPrimitiveReverse::writeSlopeForNetArc()` now picks the active
  net-delay candidate by wire delay and the active net-slew candidate by load
  slew. Their root/input-slew finite-difference slopes are stored separately.
- Added `RouteGradActiveGateSlewWinnerSlope` to reconstruct the gate output
  slew winner by scanning gate backward arcs and recomputing DMP gate driver
  waves. Gate slew reverse no longer depends on the active AT winner.
- Host reverse now accumulates `bar_pin_slew` through the reconstructed gate
  slew winner and the split net slew winner, while AT reverse still follows the
  saved active AT predecessor.
- Normal DMP timing/power kernels remain untouched; these arrays and kernels are
  only allocated/launched by the debug route-gradient APIs.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- build/install: `cmake --build . -j8` passed, then `make install` installed
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- Python import in conda `gnn` loads the installed module and exposes
  `compute_dmp_route_segment_soft_timing_grad` and
  `debug_dmp_route_segment_grad_fd_validate`.
- smoke: edge grads `1336274`, nonzero `536726`, max abs `0.101390222`;
  node grads `1460175`, nonzero `765591`, max abs `0.00375890676`;
  edge-cap grads nonzero `709551`, max abs `0.00367525133`.
- known adjoints: `edge 1014260=0.00467197`,
  `edge 1014259=0.00465100`, `edge 1014264=0.00162126`,
  `edge 942935=0.000210208`, `node 1105069=0.00274558`.
- centered FD spot check: `node 1105069` FD `0.00297885` vs adjoint
  `0.00274558`; known edge FD remains noisy and can disagree in magnitude or
  sign due global winner/branch changes.
- random-net validator smoke with `sample_net_count=5` and seed `7` produced
  5 valid edge FD rows and 5 valid node FD rows. The sampled nets were mostly
  non-critical, so adjoints were near zero while FD showed numerical/global
  branch noise; use this as an interface smoke, not as quality evidence.

Remaining gaps after slew-winner split:

- Gate primitive and net driver-wave primitive still use local finite
  differences around DMP's current branch. Replace these with analytic reverse
  from `22_dmp_pi_rc_derivative_chain.md`, especially `J_x^T mu = bar_x` and
  crossing reverse `bar_w -= bar_T V_w/V_T`.
- Global FD edge checks are not a strict oracle because perturbing one segment
  can switch DMP route/timing winners. Add fixed-winner local checks before
  judging edge-res adjoints numerically.
- Full `sample_net_count=10000` FD validation is implemented as an API but was
  not run here; it would require many global DMP reruns and should be scheduled
  as an overnight/cluster job, not a normal edit-cycle smoke.


## 2026-06-04 Update: Analytic Fanout Crossing/Wave Chain

Implemented the first analytic replacement inside the net driver-wave chain:

- Added `RouteGradWaveParamSlopes`, `RouteGradDelaySlewWaveSlopes`, and
  `RouteGradNetDriverWaveEval` in the isolated route_grad sidecar header.
- Added analytic explicit partials for driver waveform value wrt
  `(t0, dt, k0, k1, k2, k3, k4, p1, p2)`.
- Added analytic explicit partials for sink/load waveform value wrt the same
  driver-wave parameters, plus the existing Elmore derivative path.
- Added implicit crossing reverse for driver and load crossings:
  `dT/dq = -V_q/V_T` and `dL/dq = -V_q/V_L`.
- Added `delaySlewWaveParamSlopes()` for fanout sink delay/slew:
  - raw net delay: `L_vth - T_vth`,
  - raw net slew: `(L_vh - L_vl) / derate`,
  - DMP clamp behavior for negative net delay and load slew below driver slew,
  - threshold-adjust transform applied per wave parameter.
- Added `netDriverPrimitiveWaveChainSlopes()` and made net primitive reverse try
  it before the old full-output FD fallback. This keeps debug behavior robust
  while replacing the fanout crossing/wave part of the derivative chain.

Important boundary:

- Root parameter to wave parameter derivatives are still sampled locally by
  finite differences on `(C1, C2, rpi, input_slew)` and then multiplied by the
  analytic crossing/wave Jacobian. The remaining analytic step is the PI solve
  reverse from `22_dmp_pi_rc_derivative_chain.md`: `J_x^T mu = bar_x`, plus LUT
  partials for `rd`, `ceff`, and gate delay.
- Gate primitive output slew still uses the older primitive FD path. The same
  driver-crossing wave partials added here can be reused there once root PI
  reverse is added.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- `cmake --build . -j8` passed and `make install` installed the Python module.
- Python import in conda `gnn` loads
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- smoke: edge grads `1336274`, finite `1336274`, nonzero `542790`, max abs
  `0.101343098`; node grads `1460175`, finite `1460175`, nonzero `774390`, max
  abs `0.00348166737`; edge-cap nonzero `717716`, max abs `0.00347967524`.
- known adjoints: `edge 1014260=0.00376595`,
  `edge 1014259=0.00423867`, `edge 1014264=0.00159170`,
  `edge 942935=0.000212830`, `node 1105069=0.00269061`.
- centered FD spot check: `node 1105069` FD `0.00298932` vs adjoint
  `0.00269061`. Edge FD remains noisy/global-winner sensitive:
  `edge 1014260` FD `0.0133241` vs adjoint `0.00376595`,
  `edge 942935` FD `-0.00325804` vs adjoint `0.00021283`.

Next analytic target:

- Replace root-to-wave sampling with the implicit PI solve reverse. The forward
  equations and current Newton Jacobian are in `DmpGateEval.cu` around
  `findDriverParamsLocalPi()`: solve `F(t0, dt, ceff, C1, C2, rpi, rd, S)=0`,
  use `J_x^T mu = bar_x`, then reverse `F_P`, `rd`, LUT delay/slew, and PI
  coefficient algebra to obtain `(bar_C1, bar_C2, bar_rpi, bar_input_slew)`.


## 2026-06-04 Update: Analytic LUT/Rd/PI Coefficient Slopes

Extended the isolated route_grad analytic sidecar for the root-to-wave part of
PI net driver reverse:

- Added `RouteGradLutSlopes` and a route_grad-local bilinear LUT derivative
  helper matching `GPULutAllocator::gateLutWithMeta()` bin/clamp behavior.
- Added gate arc delay/slew local slopes wrt input slew and load cap.
- Added analytic `estimateRdWithSlopes()` for the DMP driver resistance estimate:
  `rd = -log(vth) * abs(D(S,Ctot) - D(S,Ctot+dC)) / dC`, including slopes wrt
  `C1`, `C2`, and input slew.
- Added analytic PI coefficient forward-mode derivatives for
  `(k0,k1,k2,k3,k4,p1,p2)` wrt `(C1,C2,rpi,input_slew)`, including the rd
  contribution from LUT delay slopes.
- Updated `netDriverPrimitiveWaveChainSlopes()` so PI branches now use analytic
  coefficient derivatives and only sample the remaining implicit-solve outputs
  `t0`, `dt`, and direct-driving-cell extra delay. Non-PI branches and analytic
  failures still fall back to the old full primitive FD path.

Current derivative-chain boundary:

- The fanout load/driver crossing and waveform coefficient algebra are now
  expanded analytically for PI net-driver candidates.
- The remaining FD inside this path is the PI solve variable part:
  `(C1,C2,rpi,input_slew) -> (t0, dt, ceff)`, plus the direct-driving-cell
  extra delay through `ceff`. This is the next `J_x^T mu = bar_x` target.
- Gate primitive slopes still use the older local FD primitive and can reuse the
  same LUT/rd/coefficient/crossing helpers after the PI solve reverse is added.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- `cmake --build . -j8` passed; the only remaining compile warning was the
  pre-existing `json.hpp` warning. `make install` installed the Python module.
- Python import in conda `gnn` loads
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- smoke: edge grads `1336274`, finite `1336274`, nonzero `542823`, max abs
  `0.101339660`; node grads `1460175`, finite `1460175`, nonzero `774443`, max
  abs `0.00348166522`; edge-cap nonzero `717762`, max abs `0.00347962874`.
- known adjoints: `edge 1014260=0.00376108`,
  `edge 1014259=0.00424414`, `edge 1014264=0.00159137`,
  `edge 942935=0.000212841`, `node 1105069=0.00269061`.
- centered FD spot check: `node 1105069` FD `0.00298934` vs adjoint
  `0.00269061`. Edge FD remains global winner/branch noisy:
  `edge 1014260` FD `0.0133241` vs adjoint `0.00376108`,
  `edge 942935` FD `-0.00325797` vs adjoint `0.000212841`.

Next analytic target:

- Implement transpose implicit solve for `findDriverParamsLocalPi()`:
  reconstruct final `F0/F1/F2` and `J_x`, solve `J_x^T mu = bar_(t0,dt,ceff)`,
  and reverse `F_P` into `bar_C1`, `bar_C2`, `bar_rpi`, `bar_rd`, and
  `bar_input_slew`. Then reverse `rd` and LUT `D/U` slopes using the helpers
  added in this update.

## 2026-06-04 Update: PI Implicit Solve Direction Slopes

Extended the isolated route_grad analytic sidecar through the remaining PI
solve variables in the net driver-wave path:

- Added `RouteGradPiCoeffSlopes` and `RouteGradPiSolveSlopes` to keep
  coefficient/current and implicit-solve directional variables separate.
- Added route_grad-local finite-ramp cap response derivatives wrt `rd`, and a
  PI-current parameter-side derivative helper for `F0 = Ipi - Iceff`.
- Added analytic PI coefficient direction slopes for `(k0..p2)` plus current
  coefficients `(A,B,D)` wrt a single direction `(dC1,dC2,drpi,drd)`.
- Added `piImplicitSolveDirectionSlopes()`: for each direction, it forms the
  current DMP PI solve Jacobian `J_x` at `(t0,dt,ceff)` and solves
  `dx = -J_x^{-1} F_p` for `(dt0, ddt, dceff)`.
- Updated `netDriverPrimitiveWaveChainSlopes()` so PI net-driver candidates no
  longer sample `t0`, `dt`, or direct-driving-cell extra delay by finite
  difference. The chain now uses analytic direction slopes for:
  `LUT/Rd -> PI coefficients/current -> implicit PI solve -> driver/load
  crossing -> net delay/slew`.
- Non-PI branches and analytic failures still fall back to the old primitive FD
  path. Active gate primitive slopes are still the older FD implementation.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- `cmake --build . -j8` passed; `make install` installed
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- Python import check loaded the installed module and exposed both route_grad
  debug APIs.
- smoke: edge grads `1336274`, finite `1336274`, nonzero `542829`, max abs
  `0.101339572`; node grads `1460175`, finite `1460175`, nonzero `774452`, max
  abs `0.00334726665`; edge-cap grads nonzero `717770`, max abs
  `0.00334591827`.
- known adjoints: `edge 1014260=0.00376348925`,
  `edge 1014259=0.00424570228`, `edge 1014264=0.00159143421`,
  `edge 942935=0.000212841245`, `node 1105069=0.00268966966`.
- centered FD spot check: `node 1105069` FD `0.00298939426` vs adjoint
  `0.00268966966`. Known edge FD remains global-winner/branch noisy:
  `edge 1014260` FD `0.0133253636` vs adjoint `0.00376348925`,
  `edge 1014259` FD `0.0174805808` vs adjoint `0.00424570228`,
  `edge 1014264` FD `-5.00e-7` vs adjoint `0.00159143421`,
  `edge 942935` FD `-0.00325767326` vs adjoint `0.000212841322`.

Remaining gaps:

- Active gate delay/slew primitive slopes still use local FD and should be
  converted to the same PI implicit solve plus driver-crossing derivative.
- The current implicit solve uses DMP's existing PI solve Jacobian branch and
  `ceff_time` clamp behavior. This is branch-local and intentionally ignores
  discrete LUT-bin, clamp, algorithm, and winner switches.
- Full `sample_net_count=10000` validation remains an overnight/cluster job; the
  edit-cycle validation above only checks ariane smoke and a few fixed ids.

## 2026-06-04 Update: Active Gate PI Primitive Analytic Slopes

Extended the same PI implicit-solve machinery to active gate primitive slopes:

- Added `driverOutputSlewWaveParamSlopes()` for driver output slew crossing
  derivatives wrt `(t0,dt,k0..p2)`.
- Added `gatePrimitiveWaveChainSlopes()` and changed active-gate delay/slew
  winner sidecar calls to try analytic PI slopes before the old local FD
  fallback.
- Corrected `piImplicitSolveDirectionSlopes()` to use the true residual
  Jacobian for the branch-local equations rather than DMP Newton's approximate
  Jacobian:
  - row 0 now differentiates `F0 = Ipi - Iceff` directly for `dt` and `ceff`,
    including `ceff_time` clamp derivatives;
  - rows 1/2 include Liberty delay/slew load-slope contributions to
    `t_vth(ceff)` and `t_vl(ceff)`.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- `cmake --build . -j8` passed; only the pre-existing `json.hpp` warning and
  nvlink power-kernel stack-size warnings appeared. `make install` installed the
  Python module.
- smoke: edge grads `1336274`, finite `1336274`, nonzero `544674`, max abs
  `0.101333111`; node grads `1460175`, finite `1460175`, nonzero `777150`, max
  abs `0.00348178908`; edge-cap grads nonzero `720266`, max abs
  `0.00347980259`.
- known adjoints: `edge 1014260=0.00396934300`,
  `edge 1014259=0.00423307814`, `edge 1014264=0.00160060216`,
  `edge 942935=0.000212860620`, `node 1105069=0.00269042788`.
- centered FD spot check: `node 1105069` FD `0.00298935865` vs adjoint
  `0.00269042788`. Known edge FD remains global-winner/branch noisy:
  `edge 1014260` FD `0.0133394820` vs adjoint `0.00396934300`,
  `edge 1014259` FD `0.0174774393` vs adjoint `0.00423307814`,
  `edge 1014264` FD `0.000188965` vs adjoint `0.00160060216`,
  `edge 942935` FD `-0.00324574321` vs adjoint `0.000212860620`.

Current analytic boundary:

- PI net-driver candidates and PI active gate primitives now use analytic
  direction slopes through LUT/Rd, PI coefficients/current, implicit PI solve,
  and driver/load crossing derivatives.
- CAP/ZERO_C2/non-smooth failures still fall back to local FD in the debug
  sidecar. Discrete algorithm, LUT-bin, clamp, and timing/slew winner switches
  remain branch-local assumptions.
- Full `sample_net_count=10000` validation remains a cluster/overnight job.

## 2026-06-04 Update: ZERO_C2 One-Pole Analytic Slopes

Extended the isolated route_grad analytic sidecar through the ZERO_C2 one-pole
branch used by DMP driver waves:

- Added `RouteGradOnePoleSolveSlopes` so one-pole implicit-solve temporaries are
  separate from PI solve variables.
- Added `zeroC2CoeffDirectionSlopes()` for the fixed-ceff one-pole coefficients
  from `initZeroC2()`.
- Added `onePoleImplicitSolveDirectionSlopes()` for the true 2x2 residual of
  `findDriverParamsLocalOnePole()`, including Liberty delay/slew load slopes and
  finite-ramp cap-response derivatives wrt `C1` and `rd`.
- Updated both `gatePrimitiveWaveChainSlopes()` and
  `netDriverPrimitiveWaveChainSlopes()` so ZERO_C2 no longer takes the primitive
  local-FD fallback when the branch-local analytic solve succeeds.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- smoke: edge grads `1336274`, finite `1336274`, nonzero `544674`, max abs
  `0.101333122`; node grads `1460175`, finite `1460175`, nonzero `777150`, max
  abs `0.00348178908`; edge-cap grads nonzero `720266`, max abs
  `0.00347980259`.
- known adjoints: `edge 1014260=0.00396934300`,
  `edge 1014259=0.00423307814`, `edge 1014264=0.00160060216`,
  `edge 942935=0.000212860643`, `node 1105069=0.00269042788`.
- centered FD spot check: `node 1105069` FD `0.00298926978` vs adjoint
  `0.00269042788`. Known edge FD remains global-winner/branch noisy:
  `edge 1014260` FD `0.0133250342` vs adjoint `0.00396934300`,
  `edge 1014259` FD `0.0174773644` vs adjoint `0.00423307814`,
  `edge 1014264` FD `0.000166612` vs adjoint `0.00160060216`,
  `edge 942935` FD `-0.00316837542` vs adjoint `0.000212860643`.

Current analytic boundary:

- PI and ZERO_C2 net-driver candidates and active gate primitives now use
  analytic direction slopes through LUT/Rd, waveform coefficients, implicit solve,
  and driver/load crossing derivatives.
- CAP fallback and non-smooth analytic failures still use local FD in the debug
  sidecar. Discrete algorithm, LUT-bin, clamp, and timing/slew winner switches
  remain branch-local assumptions.
- Full `sample_net_count=10000` validation remains a cluster/overnight job.

## 2026-06-04 Update: CAP Gate Primitive LUT Slopes

Reduced the remaining active-gate primitive local-FD fallback for the DMP CAP
fallback branch:

- `gatePrimitiveWaveChainSlopes()` now handles `DMP_ALG_CAP` by differentiating
  the fallback Liberty `capDelaySlew(C1 + C2)` directly.
- For this branch-local model, `d/dC1` and `d/dC2` share the LUT load slope,
  `d/drpi = 0`, and `d/dinput_slew` uses the LUT input-slew slope.
- This path intentionally does not create a synthetic driver waveform. CAP
  net-driver cases still cannot use load-crossing wave derivatives because the
  DMP fallback only stores table delay/slew and marks the wave as `DMP_ALG_CAP`.

Validation after build/install on `visible/ariane` with `tau_ns=0.02`:

- `cmake --build . -j8` passed with only the pre-existing `json.hpp` warning and
  nvlink power-kernel stack-size warnings. `make install` installed the updated
  `gputimer` module.
- smoke: edge grads `1336274`, finite `1336274`, nonzero `540015`, max abs
  `0.101333111`; node grads `1460175`, finite `1460175`, nonzero `777208`, max
  abs `0.00348178908`; edge-cap grads nonzero `720316`, max abs
  `0.00347980259`.
- known adjoints: `edge 1014260=0.00396934300`,
  `edge 1014259=0.00423307814`, `edge 1014264=0.00160060216`,
  `edge 942935=0.000212860621`, `node 1105069=0.00269042788`.
- centered FD spot check: `node 1105069` FD `0.00298933892` vs adjoint
  `0.00269042788`. Known edge FD remains global-winner/branch noisy:
  `edge 1014260` FD `0.0133239808` vs adjoint `0.00396934300`,
  `edge 1014259` FD `0.0174771124` vs adjoint `0.00423307814`,
  `edge 1014264` FD `0.000165486` vs adjoint `0.00160060216`,
  `edge 942935` FD `-0.00325732760` vs adjoint `0.000212860621`.

Current analytic boundary:

- Active-gate primitives cover CAP, ZERO_C2, and PI branch-local derivatives.
- Net-driver candidates cover ZERO_C2 and PI branch-local derivatives. CAP
  net-driver candidates remain outside the smooth wave-chain model because no
  driver waveform exists in the DMP CAP fallback.
- Discrete algorithm, LUT-bin, clamp, and timing/slew winner switches remain
  branch-local assumptions. Full `sample_net_count=10000` validation remains a
  cluster/overnight job.

## 2026-06-04 Update: CAP Net-Driver Table Slopes and Coverage Stats

Extended the remaining DMP CAP net-driver fallback into a branch-local analytic
chain and added a small coverage counter API:

- Added `netDriverPrimitiveCapTableSlopes()` for DMP `DMP_ALG_CAP` net-driver
  candidates. It differentiates the fallback table model used by
  `loadDelaySlewFromDriverWave()`:
  - raw net delay wrt driver-root RC is zero because the fallback delay term is
    the downstream Elmore value;
  - raw load slew follows Liberty `capDelaySlew(C1 + C2)` slew load/input-slew
    slopes;
  - `thresholdAdjust()` is chained through `thresholdAdjustedSlopes()` so load
    threshold conversion can still turn driver slew changes into delay changes;
  - direct `set_driving_cell` cases add the analytic extra-delay slope
    `table_delay(C1 + C2) - table_delay(0)`.
- Added debug-only primitive coverage stats:
  `debug_dmp_route_segment_primitive_slope_stats()` / Python wrapper
  `debug_route_segment_primitive_slope_stats()`. The normal gradient path passes
  `nullptr` and does not allocate counters; the debug path allocates one 12-entry
  `uint64` array and counts analytic, local-FD fallback, and fail hits for net
  delay, net slew, active gate, and gate slew-winner slope preparation.

Validation after build/install on `visible/ariane`:

- `cmake --build . -j8` passed; only pre-existing `json.hpp` and nvlink
  power-kernel stack-size warnings appeared. `make install` installed the updated
  `gputimer` module.
- Stats API binding loaded: `has_stats_api=True`.
- Primitive slope coverage stats:
  - `net_delay_analytic=686921`, `net_delay_fd=0`, `net_delay_fail=0`
  - `net_slew_analytic=391831`, `net_slew_fd=0`, `net_slew_fail=0`
  - `active_gate_analytic=549488`, `active_gate_fd=0`, `active_gate_fail=0`
  - `gate_slew_analytic=552500`, `gate_slew_fd=0`, `gate_slew_fail=0`
- Normal route-grad smoke still passes: edge grads `1336274`, finite `1336274`,
  nonzero `540015`, max abs `0.101333111`; node grads `1460175`, finite
  `1460175`, nonzero `777208`, max abs `0.00348178908`; edge-cap grads nonzero
  `720316`, max abs `0.00347980259`.

Current analytic boundary:

- On `visible/ariane`, all measured primitive slope-prep calls now use analytic
  chains; local primitive FD fallback is not hit.
- The FD helper functions remain as debug fallback for out-of-family cases on
  other designs. Discrete algorithm changes, LUT-bin switches, clamp switches,
  and timing/slew winner switches are still treated as branch-local assumptions.
- Full `sample_net_count=10000` FD validation has not been run because that API
  performs full DMP timing per sampled edge/node perturbation and remains a
  cluster/overnight job.

## 2026-06-04 Update: Fixed-Topology RC-Tree Local Gradcheck

Added a fixed-topology local gradcheck for the RC-tree/root moment reverse chain:

- Added `debug_dmp_route_segment_rc_tree_gradcheck()` / Python wrapper
  `debug_route_segment_rc_tree_gradcheck()`.
- This checker does not rerun timing and does not depend on timing/wave winner
  choices. It builds a deterministic synthetic scalar objective over sampled
  nets and validates only the local formulas:
  - pin Elmore delay contribution `delay_child = delay_parent + R_edge * M_child`;
  - root moment reduction `C1 = N^2/P`, `C2 = M - C1`,
    `rpi = -P^2/N^3`;
  - reverse propagation from `bar_elmore`, `bar_root_c1`, `bar_root_c2`, and
    `bar_root_rpi` to `edge_res_grad` and `node_cap_grad`.
- The finite difference perturbs one sampled edge resistance and the sampled
  child node capacitance, with the same all-corners node-cap perturbation shape
  used by the route-segment FD API.

Validation after build/install on `visible/ariane`:

- `cmake --build . -j8` passed; only pre-existing `json.hpp` and nvlink
  power-kernel stack-size warnings appeared. `make install` installed the updated
  `gputimer` module.
- API binding loaded: `has_rc_tree_gradcheck=True`.
- `sample_net_count=100`, seed `11`: all 100 edge and node rows valid;
  edge max rel `4.52e-4`, p95 rel `2.41e-4`; node max rel `5.62e-4`, p95 rel
  `4.39e-4`.
- `sample_net_count=10000`, seed `11`: all 10000 edge and node rows valid;
  edge max rel `5.67e-4`, p50 rel `9.79e-5`, p95 rel `3.07e-4`, p99 rel
  `3.90e-4`, no edge rows above `1e-3`; node max rel `1.84e-3`, p50 rel
  `1.45e-4`, p95 rel `4.33e-4`, p99 rel `5.21e-4`, one node row above
  `1e-3` and none above `1e-2`.
- Normal route-grad smoke still passes after the new checker:
  primitive stats remain FD-free on ariane
  (`net_delay_fd=0`, `net_slew_fd=0`, `active_gate_fd=0`, `gate_slew_fd=0`, all
  fail counters `0`), and output tensors are finite:
  edge grads `1336274/1336274` finite, node grads `1460175/1460175` finite,
  edge-cap grads nonzero `720316`.

Current analytic/validation boundary:

- On `visible/ariane`, primitive wave/table slope preparation is analytic-only,
  and the fixed-topology RC-tree/root moment reverse chain has a 10000-random-net
  local FD check.
- The full global `debug_dmp_route_segment_grad_fd_validate(sample_net_count=10000)`
  still has not been run. That check reruns full DMP timing per perturbation and
  is affected by timing/wave/winner switches, so it is useful as a stress test
  but not as a strict local derivative oracle.
- Local FD fallback functions are still present for out-of-family/failure cases
  on other designs, but ariane coverage shows they are not used in measured
  primitive slope-prep paths.



## 2026-06-04 Update: Cross-Design Route-Grad Validation And PI Fallback Classification

Implemented debug-only fallback classification for primitive slope stats:

- Extended `debug_dmp_route_segment_primitive_slope_stats()` with algorithm buckets for FD fallback:
  `*_fd_cap`, `*_fd_zero_c2`, `*_fd_pi`, `*_fd_other`.
- Added PI reason counters for analytic failure classification:
  `pi_fail_coeff`, `pi_fail_implicit_*`, `pi_fail_rd`, `pi_fail_init`,
  `pi_fail_forward_solve`, `pi_fail_wave_slope`, `pi_fail_extra_lut`.
- These counters reuse the existing small `uint64` stats buffer. They do not add
  per-pin/per-edge arrays and are only active when the debug stats API passes a
  non-null `primitive_stats` pointer.
- Normal `compute_dmp_route_segment_soft_timing_grad()` still uses the isolated route_grad
  sidecar and does not allocate the extra reason counters.

Build/install status:

- `cmake --build . -j8` passed.
- `make install` installed
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- Remaining warnings are pre-existing `json.hpp` missing-return and nvlink power
  kernel stack-size warnings.

Cross-design visible validation with route segments:

- `visible/ariane` final regression:
  - primitive stats: `net_delay_analytic=686920`, `net_slew_analytic=391832`,
    `active_gate_analytic=549488`, `gate_slew_analytic=552500`; no FD/fail.
  - RC-tree local gradcheck, 1000 sampled nets: edge/node valid `1000/1000`;
    edge max relative error `5.718979649855304e-4`, node max relative error
    `5.621474689915232e-4`; no sample above `1e-3`.
- `visible/NV_NVDLA_partition_c`:
  - primitive stats: all analytic; no FD/fail.
  - RC-tree local gradcheck, 1000 sampled nets: edge max rel
    `5.341355479136763e-4`, node max rel `5.588978562622569e-4`; no sample
    above `1e-3`.
- `visible/mempool_tile_wrap`:
  - primitive stats: all analytic; no FD/fail.
  - RC-tree local gradcheck, 1000 sampled nets: edge max rel
    `4.520877825914885e-4`, node max rel `5.873959769204265e-4`; no sample
    above `1e-3`.
- `visible/bsg_chip`:
  - primitive stats: all analytic; no FD/fail.
  - RC-tree local gradcheck, 1000 sampled nets: edge max rel
    `5.11828654236461e-4`, node max rel `5.417944239713039e-4`; no sample
    above `1e-3`.
- `visible/mempool_group`:
  - route segment file size is about `1.4G`; design has about `12.0M` pins.
  - primitive stats: `net_delay_fd=11`, `net_slew_fd=3`,
    `active_gate_fd=5`, `gate_slew_fd=3`, all classified as PI; fail counters
    are zero.
  - PI reason stats: `pi_fail_forward_solve=22`, exactly matching the total FD
    fallback count. This means the remaining fallback is not a missing CAP/ZERO/PI
    derivative branch; it is the derivative sidecar re-running
    `findDriverParamsLocalPi()` and hitting the same forward solver boundary on
    a tiny number of PI primitives. `DmpDriverWave` does not save `ceff`, so
    removing this last fallback would require either saving PI `ceff` in the
    forward wave/model state or adding a sidecar inverse-LUT/recovery path. That
    would touch the main DMP data model and was intentionally not done here.
  - RC-tree local gradcheck, 1000 sampled nets: edge max rel
    `5.574966154639001e-4`, node max rel `5.734502222859804e-4`; no sample
    above `1e-3`.

Cluster boundary:

- `visible/mempool_cluster.route_segments` is about `5.1G`, roughly 3.6x the
  `mempool_group` route file. It was not run in this edit-cycle validation.
  Given group has about `12.0M` pins and already exercises the full primitive
  stats path, cluster should be treated as a scheduled large-case run rather
  than a normal quick smoke.

Implementation interpretation:

- The analytic chain now covers CAP, ZERO_C2, and PI for active gate, gate slew
  winner, net delay winner, and net slew winner.
- The local fixed-topology RC-tree reverse is numerically aligned to FD across
  small, medium, and large visible cases.
- The only observed primitive fallback is a tiny PI forward-solver boundary on
  `mempool_group`; the fallback is local FD, not failure, and it is explicitly
  classified by the debug stats API.


## 2026-06-04 Update: PI Rd Forward-Consistency And Ceff Recovery

Closed the remaining `mempool_group` primitive FD fallback observed after the
first cross-design validation. Root cause was not a missing analytic derivative
branch; it was a forward-consistency issue in the sidecar PI setup:

- DMP forward `DmpGateArcMeta::estimateRd()` computes `rd` using float-cast
  `cap1`, float-cast `cap_delta`, float-cast LUT delays, and denominator
  `cap2_f - cap1_f`.
- The route_grad sidecar previously computed the same `rd` value in double using
  the nominal double `cap_delta`. On boundary PI cases this tiny value mismatch
  was enough for the second `findDriverParamsLocalPi()` call inside the analytic
  sidecar to fail even though the forward wave had already succeeded.
- Updated `RouteGradNetPrimitiveReverse::estimateRdWithSlopes()` so the `rd`
  value is forward-consistent with DMP while retaining the continuous LUT slope
  approximation for `rd_c1`, `rd_c2`, and `rd_input_slew`.
- Added sidecar-only `routeGradRecoverCeffFromGateDelay()`: if the analytic PI
  path still cannot re-solve `(t0, dt, ceff)`, it recovers `ceff` by bisection
  on the same `DmpGateArcMeta::capDelaySlew()` delay LUT over `[0, C1+C2]`,
  using the already successful forward gate delay. This does not touch the main
  `DmpDriverWave` or DMP model state.
- Added debug stat `pi_recovered_ceff_from_delay` to make this recovery visible.

Validation after build/install:

- `cmake --build . -j8` passed.
- `make install` installed the updated Python module.
- `visible/mempool_group` primitive stats after the fix:
  - `net_delay_analytic=19181863`
  - `net_slew_analytic=11015567`
  - `active_gate_analytic=13990244`
  - `gate_slew_analytic=13992540`
  - no FD/fail counters are nonzero
  - `pi_recovered_ceff_from_delay=1`
- `visible/ariane` regression after the fix:
  - primitive stats all analytic: `net_delay_analytic=686922`,
    `net_slew_analytic=391829`, `active_gate_analytic=549488`,
    `gate_slew_analytic=552500`; no FD/fail.
  - RC-tree local gradcheck, 1000 sampled nets: edge/node valid `1000/1000`;
    edge max relative error `5.718979649855304e-4`, node max relative error
    `5.621474689915232e-4`; no sample above `1e-3`.

Current status at this point, later superseded below:

- This checkpoint had only tested through `mempool_group`. Subsequent sections
  include the later `mempool_cluster` coverage run and the analytic-only main
  gradient path cleanup.

## 2026-06-04 Update: Large-Case Primitive Coverage

After the forward-consistent `estimateRdWithSlopes()` fix and route_grad-local
ceff recovery from the forward gate delay, the analytic primitive coverage was
checked on larger ISPD2025 visible cases. No FD fallback or primitive failure
counters were reported.

- `visible/ariane`: `net_delay_analytic=686922`, `net_slew_analytic=391829`,
  `active_gate_analytic=549488`, `gate_slew_analytic=552500`.
- `visible/mempool_group`: `net_delay_analytic=19181863`,
  `net_slew_analytic=11015567`, `active_gate_analytic=13990244`,
  `gate_slew_analytic=13992540`, `pi_recovered_ceff_from_delay=1`.
- `visible/mempool_cluster`: `net_delay_analytic=69869575`,
  `net_slew_analytic=39320576`, `active_gate_analytic=50747656`,
  `gate_slew_analytic=50843040`, `pi_recovered_ceff_from_delay=110`.

Local fixed-topology RC-tree gradcheck was also run on `visible/ariane` with
1000 sampled nets after the analytic primitive chain was installed: edge and
node samples were all valid, p95 relative error was below `5e-4`, and max
relative error stayed below `6e-4`. An earlier 10000-net ariane RC-tree check
covered the requested random-net interface; edge samples stayed below about
`5.7e-4` relative error, while node samples had one near-zero/noisy sample above
`1e-3` but none above `1e-2`.

## 2026-06-04 Update: Main Gradient Path Is Analytic-Only

Cleaned the route_grad sidecar so primitive local finite-difference fallback is
no longer called from the main route-segment gradient path. The debug FD helper
functions remain in the sidecar source for standalone diagnostics, but `rg` now
shows no `fd_ok` or primitive `*FiniteDiff()` call sites from the slope kernels;
only the function declarations/bodies remain. Analytic failure now increments
the corresponding `*_fail` counter instead of silently substituting a local FD
slope. This keeps the implementation boundary clear: finite difference is only
for validation/debug, not part of the derivative chain.

Build/install after this cleanup:

- `cmake --build . -j8` passed. New route_grad unused-constant warnings were
  removed; remaining warnings were pre-existing `json.hpp` and power nvlink
  stack-size warnings.
- `make install` installed
  `cpp_to_py/cpybin/gputimer.cpython-310-x86_64-linux-gnu.so`.
- Python import confirmed the installed module exposes route_grad debug APIs.

Post-cleanup validation:

- `visible/ariane` primitive coverage: `net_delay_analytic=686922`,
  `net_slew_analytic=391831`, `active_gate_analytic=549488`,
  `gate_slew_analytic=552500`, and `FD_FAIL_NONZERO {}`.
- `visible/ariane` RC-tree local gradcheck with 1000 sampled nets: edge/node
  valid `1000/1000`; edge max relative error `4.634e-4`, p95 `3.040e-4`;
  node max relative error `5.874e-4`, p95 `4.312e-4`; no sample above `1e-3`.
- `visible/mempool_group` primitive coverage: `net_delay_analytic=19181861`,
  `net_slew_analytic=11015568`, `active_gate_analytic=13990244`,
  `gate_slew_analytic=13992540`, `pi_recovered_ceff_from_delay=1`, and
  `FD_FAIL_NONZERO {}`.
- `visible/mempool_cluster` primitive coverage: `net_delay_analytic=69869569`,
  `net_slew_analytic=39320569`, `active_gate_analytic=50747656`,
  `gate_slew_analytic=50843040`, `pi_recovered_ceff_from_delay=110`, and
  `FD_FAIL_NONZERO {}`.

Current completion interpretation for the derivative-chain goal: within the
fixed-topology/fixed-winner/non-smooth branch assumptions documented in
`22_dmp_pi_rc_derivative_chain.md`, the route-segment gradient chain is now
expanded analytically through endpoint soft objective, active AT/RAT/slew
reverse, gate/net primitive delay/slew, LUT/Rd, CAP/ZERO_C2/PI implicit solves,
driver/load fanout crossing wave derivatives, threshold adjustment, root PI
moment reverse, sink Elmore reverse, and segment R/C adjoints. Global FD remains
a noisy validation tool because route/timing winner and DMP branch switches are
real non-smooth points.

