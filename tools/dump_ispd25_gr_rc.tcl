proc getenv_default {name default_value} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default_value
}

proc require_file {path label} {
  if {![file exists $path]} {
    error "Missing $label: $path"
  }
}

set bench_root [getenv_default BENCH_ROOT "/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks"]
set design_set [getenv_default DESIGN_SET visible]
set design_name [getenv_default DESIGN_NAME ""]
set route_file [getenv_default ROUTE_FILE ""]
set gr_rc_out [getenv_default GR_RC_OUT ""]

if {$design_name eq ""} {
  error "DESIGN_NAME is required"
}
if {$route_file eq ""} {
  error "ROUTE_FILE is required"
}
if {$gr_rc_out eq ""} {
  error "GR_RC_OUT is required"
}

set design_dir [file join $bench_root $design_set $design_name]
set def_file [file join $design_dir ${design_name}.def]
set sdc_file [file join $design_dir ${design_name}.sdc]
set rc_file [file join $bench_root NanGate45 setRC.tcl]
set libdir [file join $bench_root NanGate45 lib]
set lefdir [file join $bench_root NanGate45 lef]

set lib_files [list \
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
  [file join $libdir fakeram45_64x62.lib] \
  [file join $libdir fakeram45_64x64.lib] \
  [file join $libdir fakeram45_64x124.lib] \
]

set lef_files [list \
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
  [file join $lefdir fakeram45_64x62.lef] \
  [file join $lefdir fakeram45_64x64.lef] \
  [file join $lefdir fakeram45_64x124.lef] \
]

require_file $def_file DEF
require_file $sdc_file SDC
require_file $route_file route_segments
require_file $rc_file RC_setup
foreach lib_file $lib_files {
  require_file $lib_file liberty
}
foreach lef_file $lef_files {
  require_file $lef_file LEF
}

file mkdir [file dirname $gr_rc_out]

puts "===== ISPD25_GR_RC_DUMP_BEGIN ====="
puts "bench_root: $bench_root"
puts "design_set: $design_set"
puts "design_name: $design_name"
puts "route_file: $route_file"
puts "gr_rc_out: $gr_rc_out"

foreach lib_file $lib_files {
  read_liberty $lib_file
}
foreach lef_file $lef_files {
  read_lef $lef_file
}

read_def $def_file
read_sdc $sdc_file
source $rc_file

if {[llength [info commands sta::set_crpr_enabled]]} {
  sta::set_crpr_enabled 0
}
set ::sta_crpr_enabled 0
set sta_report_default_digits 8
catch {
  set_cmd_units -time ns -capacitance pF -current mA -voltage V \
    -resistance Ohm -distance um -power mW -digits 8
}
catch {set_units -power mW}

read_global_route_segments $route_file
estimate_parasitics -global_routing
report_tns
report_wns
my_dump_gr_rc -outfile $gr_rc_out

puts "gr_rc_tsv: $gr_rc_out"
puts "===== ISPD25_GR_RC_DUMP_END ====="
