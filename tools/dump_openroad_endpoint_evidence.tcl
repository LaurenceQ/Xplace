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

proc csv_escape {value} {
  set text [string map {"\"" "\"\""} $value]
  return "\"$text\""
}

proc prop_or_empty {obj prop_name} {
  if {[catch {set value [get_property $obj $prop_name]}]} {
    return ""
  }
  return $value
}

proc normalize_pin_name {pin_name} {
  # Xplace prints hierarchical pins as inst:pin; OpenROAD expects inst/pin.
  set colon [string last ":" $pin_name]
  if {$colon >= 0} {
    return [string replace $pin_name $colon $colon "/"]
  }
  return $pin_name
}

set bench_root [getenv_default BENCH_ROOT "/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks"]
set design_set [getenv_default DESIGN_SET visible]
set design_name [getenv_default DESIGN_NAME ""]
set segment_in [getenv_default SEGMENT_IN ""]
set out_dir [getenv_default OUT_DIR ""]
set selected_pins_raw [getenv_default SELECT_PINS ""]
set max_fanout_skip [getenv_default MAX_FANOUT_SKIP 300]

if {$design_name eq ""} {
  error "DESIGN_NAME is required"
}
if {$out_dir eq ""} {
  error "OUT_DIR is required"
}

set design_dir [file join $bench_root $design_set $design_name]
set def_file [file join $design_dir ${design_name}.def]
set sdc_file [file join $design_dir ${design_name}.sdc]
set rc_file [file join $bench_root NanGate45 setRC.tcl]
set libdir [file join $bench_root NanGate45 lib]
set lefdir [file join $bench_root NanGate45 lef]

if {$segment_in eq ""} {
  set segment_in [file join $bench_root openroad_gr_segments_skip_fanout${max_fanout_skip} $design_set ${design_name}.route_segments]
}

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
require_file $segment_in route_segments
require_file $rc_file RC_setup
foreach lib_file $lib_files {
  require_file $lib_file liberty
}
foreach lef_file $lef_files {
  require_file $lef_file LEF
}

file mkdir $out_dir

puts "===== OPENROAD_ENDPOINT_EVIDENCE_BEGIN ====="
puts "bench_root: $bench_root"
puts "design_set: $design_set"
puts "design_name: $design_name"
puts "def_file: $def_file"
puts "sdc_file: $sdc_file"
puts "segment_in: $segment_in"
puts "out_dir: $out_dir"

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

read_global_route_segments $segment_in
estimate_parasitics -global_routing
report_tns
report_wns

set worst_checks [file join $out_dir openroad_worst_checks.rpt]
report_checks -fields {input_pin slew capacitance net} -endpoint_count 200 -sort_by_slack > $worst_checks
puts "worst_checks: $worst_checks"

set selected_checks [file join $out_dir openroad_selected_checks.rpt]
set selected_fp [open $selected_checks w]
puts $selected_fp "Selected pins: $selected_pins_raw"
foreach raw_pin [split $selected_pins_raw "|"] {
  if {$raw_pin eq ""} {
    continue
  }
  set pin_name [normalize_pin_name $raw_pin]
  puts $selected_fp "\n===== SELECTED_PIN $raw_pin -> $pin_name ====="
  set pins [get_pins -quiet $pin_name]
  if {[llength $pins] == 0} {
    puts $selected_fp "MISSING_PIN $pin_name"
    continue
  }
  set pin [lindex $pins 0]
  puts $selected_fp "full_name: [get_full_name $pin]"
  puts $selected_fp "slack_max: [prop_or_empty $pin slack_max]"
  puts $selected_fp "slack_max_rise: [prop_or_empty $pin slack_max_rise]"
  puts $selected_fp "slack_max_fall: [prop_or_empty $pin slack_max_fall]"
  puts $selected_fp "arrival_max_rise: [prop_or_empty $pin arrival_max_rise]"
  puts $selected_fp "arrival_max_fall: [prop_or_empty $pin arrival_max_fall]"
  puts $selected_fp "slew_max: [prop_or_empty $pin slew_max]"
  puts $selected_fp "slew_max_rise: [prop_or_empty $pin slew_max_rise]"
  puts $selected_fp "slew_max_fall: [prop_or_empty $pin slew_max_fall]"
}
close $selected_fp
foreach raw_pin [split $selected_pins_raw "|"] {
  if {$raw_pin eq ""} {
    continue
  }
  set pin_name [normalize_pin_name $raw_pin]
  set pins [get_pins -quiet $pin_name]
  if {[llength $pins] == 0} {
    continue
  }
  set pin [lindex $pins 0]
  report_checks -to $pin -fields {input_pin slew capacitance net} -endpoint_count 5 -sort_by_slack >> $selected_checks
}
puts "selected_checks: $selected_checks"

array unset endpoint_kind
set endpoints {}

proc add_endpoint_pins {pins kind} {
  global endpoints endpoint_kind
  foreach pin $pins {
    set name [get_full_name $pin]
    if {![info exists endpoint_kind($name)]} {
      lappend endpoints $pin
      set endpoint_kind($name) $kind
    } elseif {[string first $kind $endpoint_kind($name)] < 0} {
      append endpoint_kind($name) "|$kind"
    }
  }
}

catch {add_endpoint_pins [all_registers -data_pins] data}
catch {add_endpoint_pins [all_registers -async_pins] async}
catch {add_endpoint_pins [all_outputs] output}

set endpoint_csv [file join $out_dir openroad_endpoint_slacks.csv]
set fp [open $endpoint_csv w]
puts $fp "pin_name,kind,lib_pin_name,direction,clock_domains,slack_max,slack_max_rise,slack_max_fall,arrival_max_rise,arrival_max_fall,slew_max,slew_max_rise,slew_max_fall,capacitance"
foreach pin $endpoints {
  set name [get_full_name $pin]
  set row [list \
    [csv_escape $name] \
    [csv_escape $endpoint_kind($name)] \
    [csv_escape [prop_or_empty $pin lib_pin_name]] \
    [csv_escape [prop_or_empty $pin direction]] \
    [csv_escape [prop_or_empty $pin clock_domains]] \
    [prop_or_empty $pin slack_max] \
    [prop_or_empty $pin slack_max_rise] \
    [prop_or_empty $pin slack_max_fall] \
    [prop_or_empty $pin arrival_max_rise] \
    [prop_or_empty $pin arrival_max_fall] \
    [prop_or_empty $pin slew_max] \
    [prop_or_empty $pin slew_max_rise] \
    [prop_or_empty $pin slew_max_fall] \
    [prop_or_empty $pin capacitance] \
  ]
  puts $fp [join $row ","]
}
close $fp
puts "endpoint_csv: $endpoint_csv"
puts "endpoint_count: [llength $endpoints]"
puts "===== OPENROAD_ENDPOINT_EVIDENCE_END ====="
