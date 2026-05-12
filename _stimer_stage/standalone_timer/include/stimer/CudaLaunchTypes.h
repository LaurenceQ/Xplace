#pragma once

namespace stimer {

struct LaunchShape {
  unsigned grid_x = 1;
  unsigned block_x = 1;
};

struct CellDelayKernelArgs {
  int num_arcs = 0;
  const int* arc_from_pin = nullptr;
  const int* arc_to_pin = nullptr;
  const int* arc_table_id = nullptr;
  const float* pin_slew = nullptr;
  const float* pin_arrival = nullptr;
  float* arc_delay = nullptr;
  float* arc_output_slew = nullptr;
};

struct RcDelayKernelArgs {
  int num_driver_arcs = 0;
  const int* driver_arc_to_net = nullptr;
  const int* net_sink_begin = nullptr;
  const int* net_sink_end = nullptr;
  const float* rc_edge_res = nullptr;
  const float* rc_node_cap = nullptr;
  float* load_delay = nullptr;
  float* load_slew = nullptr;
};

struct PropagationKernelArgs {
  int num_pins = 0;
  int num_arcs = 0;
  const int* level_begin = nullptr;
  const int* level_end = nullptr;
  float* pin_arrival = nullptr;
  float* pin_required = nullptr;
  float* pin_slew = nullptr;
};

struct TimingPropagationLaunch {
  LaunchShape shape;
  CellDelayKernelArgs cell;
  RcDelayKernelArgs rc;
  PropagationKernelArgs prop;
};

}  // namespace stimer
