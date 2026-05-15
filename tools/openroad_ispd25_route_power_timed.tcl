# Timed OpenROAD eval for saved ISPD2025 route segments.
#
# Required env:
#   DESIGN_NAME
# Optional env:
#   DESIGN_SET, BENCH_ROOT, SEGMENT_IN

proc stage {name body} {
  set t0 [clock microseconds]
  uplevel 1 $body
  set t1 [clock microseconds]
  puts [format "OR_STAGE %s %.6f" $name [expr {($t1 - $t0) / 1000000.0}]]
}

set script_dir [file dirname [file normalize [info script]]]
set bench_root [expr {[info exists ::env(BENCH_ROOT)] && $::env(BENCH_ROOT) ne "" ? $::env(BENCH_ROOT) : [file dirname $script_dir]}]
set design_set [expr {[info exists ::env(DESIGN_SET)] && $::env(DESIGN_SET) ne "" ? $::env(DESIGN_SET) : "visible"}]
if {![info exists ::env(DESIGN_NAME)] || $::env(DESIGN_NAME) eq ""} {
  error "DESIGN_NAME is required"
}
set design_name $::env(DESIGN_NAME)
set design_dir [file join $bench_root $design_set $design_name]
set def_file [file join $design_dir ${design_name}.def]
set sdc_file [file join $design_dir ${design_name}.sdc]
set segment_in [expr {[info exists ::env(SEGMENT_IN)] && $::env(SEGMENT_IN) ne "" ? $::env(SEGMENT_IN) : [file join $bench_root openroad_gr_segments_skip_fanout300 $design_set ${design_name}.route_segments]}]
set libdir [file join $bench_root NanGate45 lib]
set lefdir [file join $bench_root NanGate45 lef]

set LIB_FILES [list \
  [file join $libdir NangateOpenCellLibrary_typical.lib] \
  [file join $libdir fakeram45_256x16.lib] \
  [file join $libdir fakeram45_256x32.lib] \
  [file join $libdir fakeram45_256x64.lib] \
  [file join $libdir fakeram45_32x32.lib] \
  [file join $libdir fakeram45_128x256.lib] \
  [file join $libdir fakeram45_128x116.lib] \
  [file join $libdir fakeram45_128x32.lib] \
  [file join $libdir fakeram45_256x48.lib] \
  [file join $libdir fakeram45_512x64.lib] \
  [file join $libdir fakeram45_64x256.lib] \
  [file join $libdir fakeram45_64x62.lib] \
  [file join $libdir fakeram45_64x64.lib] \
  [file join $libdir fakeram45_64x124.lib] \
]
set LEF_FILES [list \
  [file join $lefdir NangateOpenCellLibrary.tech.lef] \
  [file join $lefdir NangateOpenCellLibrary.macro.mod.lef] \
  [file join $lefdir fakeram45_256x16.lef] \
  [file join $lefdir fakeram45_256x32.lef] \
  [file join $lefdir fakeram45_256x64.lef] \
  [file join $lefdir fakeram45_32x32.lef] \
  [file join $lefdir fakeram45_128x256.lef] \
  [file join $lefdir fakeram45_128x116.lef] \
  [file join $lefdir fakeram45_128x32.lef] \
  [file join $lefdir fakeram45_256x48.lef] \
  [file join $lefdir fakeram45_512x64.lef] \
  [file join $lefdir fakeram45_64x256.lef] \
  [file join $lefdir fakeram45_64x62.lef] \
  [file join $lefdir fakeram45_64x64.lef] \
  [file join $lefdir fakeram45_64x124.lef] \
]

puts "===== OPENROAD_ISPD25_ROUTE_POWER_TIMED_BEGIN ====="
puts "design_set: $design_set"
puts "design_name: $design_name"
puts "def_file: $def_file"
puts "sdc_file: $sdc_file"
puts "segment_in: $segment_in"

if {[info exists ::env(OR_DEBUG_LEVELIZE)] && $::env(OR_DEBUG_LEVELIZE) ne "" && $::env(OR_DEBUG_LEVELIZE) ne "0"} {
  set_debug_level STA levelize $::env(OR_DEBUG_LEVELIZE)
}
if {[info exists ::env(OR_DEBUG_POWER_ACTIVITY)] && $::env(OR_DEBUG_POWER_ACTIVITY) ne "" && $::env(OR_DEBUG_POWER_ACTIVITY) ne "0"} {
  set_debug_level STA power_activity $::env(OR_DEBUG_POWER_ACTIVITY)
}

stage read_input {
  foreach lib_file $LIB_FILES { read_liberty $lib_file }
  foreach lef_file $LEF_FILES { read_lef $lef_file }
  read_def $def_file
  read_sdc $sdc_file
  source [file join $bench_root NanGate45 setRC.tcl]
  if {[llength [info commands sta::set_crpr_enabled]]} {
    sta::set_crpr_enabled 0
  }
  set ::sta_crpr_enabled 0
}
set sta_report_default_digits 8
stage read_route_segments { read_global_route_segments $segment_in }
stage build_rc { estimate_parasitics -global_routing }
stage timer {
  report_tns
  report_wns
}
stage power { report_power }
if {[info exists ::env(OR_DUMP_POWER_CSV)] && $::env(OR_DUMP_POWER_CSV) ne ""} {
  set power_csv $::env(OR_DUMP_POWER_CSV)
  if {[info exists ::env(OR_DUMP_POWER_PINS_CSV)] && $::env(OR_DUMP_POWER_PINS_CSV) ne ""} {
    set power_pins_csv $::env(OR_DUMP_POWER_PINS_CSV)
  } else {
    set power_pins_csv [file rootname $power_csv]_pins.csv
  }
  if {[info exists ::env(OR_DUMP_POWER_ARCS_CSV)] && $::env(OR_DUMP_POWER_ARCS_CSV) ne ""} {
    set power_arcs_csv $::env(OR_DUMP_POWER_ARCS_CSV)
  } else {
    set power_arcs_csv [file rootname $power_csv]_internal_arcs.csv
  }
  if {[info exists ::env(OR_DUMP_POWER_LEAKAGE_CSV)] && $::env(OR_DUMP_POWER_LEAKAGE_CSV) ne ""} {
    set power_leakage_csv $::env(OR_DUMP_POWER_LEAKAGE_CSV)
  } else {
    set power_leakage_csv [file rootname $power_csv]_leakage.csv
  }
  file mkdir [file dirname $power_csv]
  stage power_instances {
    my_dump_power \
      -outfile $power_csv \
      -pin_outfile $power_pins_csv \
      -arc_outfile $power_arcs_csv \
      -leakage_outfile $power_leakage_csv
  }
}
puts "===== OPENROAD_ISPD25_ROUTE_POWER_TIMED_END ====="
