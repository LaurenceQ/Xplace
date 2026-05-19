# 假设与验证结果
Scope: ISPD2025 route-segment timing/power alignment. Keep <=50 lines.

## sky130 Power
- H: non-ideal SPEF flow should not use SDC ideal-clock slew override.
  Result: limiting override to `ideal_clock` fixes `blabla`/`zipdiv`.
- H: non-ideal clock seed activity must not be overwritten by propagation.
  Result: guarding overwrite by `ideal_clock` fixes `picorv32a`/`wbqspiflash`;
  sky130 21/21 Xplace-only worst total err `3.692918e-06`.

## Global Rejections
- Voltage `1.1`, DMP load alone, async clear/preset arcs: no fix.
- Timing levels, root-indeg/timing-root seeding, clock closure, loop roots:
  no useful change.
- Pass count is not a fix: 200/500/1000/2000 raises power but 2000 still low/slow.
- Broad seq seeding over-shoots: all zero-indeg seq outputs `16.2367 W`;
  feedback D-only fixed one mempool by cancellation but broke bsg.
- Expression-width rejected: NanGate45 `.lib` max function/when/seq vars `8`,
  below Xplace 16-var enum limit.

## OpenSTA Facts
- OpenSTA seeds `levelize_->roots()`, then iterates register-output activity
  up to 50 passes.
- Register outputs are enqueued when Liberty seq output function matches
  `IQ/IQN`.
## NanGate45 visible/mempool_group
- Baseline: OR total `4.589715868 W`, X total `4.494483077 W`, err `2.0749%`;
  std-cell internal/switching activity low, leakage aligned.
- H: local `a_full_q_reg/D,Q` are first source.
  Result: false; OR has earlier nonzero pins.
- H: `status_cnt` Q/QN recognition or seq pending is first root cause.
  Result: false; OR status D nonzero appears before Q.
- H: CUDA propagation queue is first root cause.
  Result: false; X CPU and CUDA probes both keep source-chain density `0`.
- H: missing OpenSTA-equivalent root/frontier seed.
  Result: false; all OR seeded roots are in X; X only extra root is `clk_i`.
- H: matched-root path propagation is missing an enqueue.
  Result: false/partial; X path reaches g99, but full final diff shows the
  first upstream break is before g99.
- H: g99 boolean eval is the first bug.
  Result: false as first cause; g99 is downstream. Full diff shows
  `FE_RC_119370_0/A1` side chain is OR-nonzero/X-zero, causing
  `FE_RC_119370_0/ZN` and then g99/A2 to become zero in X.
- H: `FE_RC_97449_0` has matching inputs but wrong output.
  Result: false; B2 is OR nonzero/X zero, so its output mismatch is downstream.
- H: strict B2 max-OR/X-zero branch reaches all-input-match gate.
  Result: false; stops at `clk_en_reg/Q`; `GN` matches, `D` is OR/X nonzero mismatch.
- H: NanGate45 latch/gated-clock activity missing.
  Result: true; X ignored latch `data_in/enable`; fix gives Ptotal err `0.1575%`.
