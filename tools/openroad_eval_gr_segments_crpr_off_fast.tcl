# Fast timed wrapper for ISPD2025 saved global-route segment eval.
#
# This sources the stable benchmark CRPR-off eval Tcl, but suppresses the
# expensive report_power/report_checks tail.  The reference WNS/TNS semantics
# are unchanged: read saved route segments, estimate global-route parasitics,
# report_tns, and report_wns with CRPR disabled by the sourced script.

if {[llength [info commands report_power]]} {
  rename report_power report_power_orig
  proc report_power {args} {
    puts "report_power_skipped_for_timed_fast_eval: 1"
  }
}

if {[llength [info commands report_checks]]} {
  rename report_checks report_checks_orig
  proc report_checks {args} {
    global checks_out
    if {[info exists checks_out] && $checks_out ne ""} {
      file mkdir [file dirname $checks_out]
      set fp [open $checks_out w]
      puts $fp "# report_checks skipped by timed fast eval wrapper"
      close $fp
    }
    puts "report_checks_skipped_for_timed_fast_eval: 1"
  }
}

source /research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks/openroad_eval_gr_segments_crpr_off.tcl
