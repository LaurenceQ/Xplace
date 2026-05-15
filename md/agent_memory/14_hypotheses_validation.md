# 假设与验证结果
Scope: ISPD2025 route-segment timing/power alignment. Keep <=50 lines.
## sky130 Power Regressions

- H: non-ideal SPEF flow should not use SDC ideal-clock slew override.
  Result: limiting override to `ideal_clock` fixes `blabla`/`zipdiv`.
- H: non-ideal clock seed activity must not be overwritten by propagation.
  Result: guarding overwrite by `ideal_clock` fixes `picorv32a`/`wbqspiflash`;
  full sky130 21/21 Xplace-only worst total err `3.692918e-06`.

## blind/mempool_group Power Gap

- Reference: OR total `4.51656961 W`; X total `4.026748111 W`.
- Timing aligns: OR/X WNS about `-1.3747767`, TNS about `-67650`.
- Diagnosis: std-cell dynamic activity is low in X; not timing or RC.

Rejected global hypotheses:
- Voltage `1.1`, DMP load alone, async clear/preset arcs: no fix.
- Timing levels, root-indeg/timing-root seeding, clock closure, loop roots:
  unchanged around `4.02674811 W`.

Pass-count hypothesis:
- `MAX_PASSES=200/500/1000/2000` -> `4.2630/4.2927/4.3014/4.3036 W`.
- At 2000, X still has `final_pending=22108`; power takes `431.6s`.
- Result: grows but remains `4.76%` low and too slow; reject as fix.

Seq-feedback hypotheses:
- Seed all zero-indeg seq outputs -> `16.2367 W`; gross over-seed.
- Seed feedback seq outputs: all `6.7335 W`, D-only `4.9013 W`.
- D-only pass25 fixes mempool by cancellation, but bsg fails `3.15%`.
- Conclusion: seq feedback matters, but broad seeding is invalid.

OpenSTA alignment facts:
- OpenSTA seeds level roots, then iterates register-output activity up to 50.
- Enqueues register outputs whose Liberty seq output function matches `IQ/IQN`.
- OR blind group: roots `5854`, roots with fanout `5461`, back edges `0`.

Tile14/chain validation:
- DEF/net connectivity is correct for prior suspects:
  `g1411:A <- g36724:ZN`, `g1410:A* <- g1414/g1413/g1411/g2268`.
- OR dynamic: `g37288:ZN 1.85012e8`, `g36724:ZN 1.85187e8`,
  `prefetch_req_vld_q_reg:Q/QN 1.91425e5`; X static.
- X chain2: `pending_refill_q_reg:D/Q/QN`, `g39183`, `g37289`,
  `g37288`, `g36724`, `prefetch_req_vld_q_reg` static after 50 passes.
- OR chain2: `g30021`, `g36733/34/36/37`, `g40707/8/9/17`,
  `g39183`, `pending_refill_q_reg` are dynamic.
- X chain3: `g620/g36725/g36763/g42326/g36765/g36750/g41386`
  are static; issue is further-upstream activity source.
- Chain4 and badpin evidence: multiple upstream scan/test/control nets are
  static in X but dynamic in OR; do not chase local power formula.
