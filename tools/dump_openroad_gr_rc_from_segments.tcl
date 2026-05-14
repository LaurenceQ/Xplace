# Dump GNNTimer/OpenROAD GR RC TSV after loading saved OpenROAD route segments.
#
# Required env:
#   DESIGN_NAME, DESIGN_SET, SEGMENT_IN, GR_RC_OUT
# Optional env:
#   BENCH_ROOT, CHECKS_OUT, MAX_FANOUT_SKIP

if {![info exists ::env(GR_RC_OUT)] || $::env(GR_RC_OUT) eq ""} {
  error "GR_RC_OUT is required"
}

set xplace_root [file dirname [file dirname [file normalize [info script]]]]
set bench_root "/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks"
if {[info exists ::env(BENCH_ROOT)] && $::env(BENCH_ROOT) ne ""} {
  set bench_root $::env(BENCH_ROOT)
}

source [file join $bench_root openroad_eval_gr_segments_crpr_off.tcl]

file mkdir [file dirname $::env(GR_RC_OUT)]
puts "gr_rc_out: $::env(GR_RC_OUT)"
my_dump_gr_rc -outfile $::env(GR_RC_OUT)
puts "gr_rc_report: $::env(GR_RC_OUT)"
