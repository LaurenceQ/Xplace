#pragma once

#include "common/lib/Liberty.h"
#include "gputimer/db/GTDatabase.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gt {

std::string normalizePowerExprString(std::string expr);
int parsePowerConstNetValue(std::string name);
LibertyCell* powerCellForNode(GTDatabase& gtdb, int node_id);
bool powerIsIoNode(GTDatabase& gtdb, int node_id);
std::pair<float, float> powerClockActivityForPin(GTDatabase& gtdb,
                                                 int pin_id,
                                                 double sdc_time_scale,
                                                 float clock_density);
void buildPowerPinNodeNetMaps(GTDatabase& gtdb,
                              int n,
                              std::vector<int>& pin_to_node,
                              std::vector<int>& pin_to_net);
void classifyPowerPins(GTDatabase& gtdb,
                       int n,
                       const std::vector<int>& pin_to_node,
                       std::vector<uint8_t>& is_load_pin,
                       std::vector<uint8_t>& is_driver_pin,
                       std::vector<uint8_t>* is_cell_pin = nullptr);
void markPowerSeqOutputPins(GTDatabase& gtdb,
                            int n,
                            const std::vector<int>& pin_to_node,
                            const std::vector<uint8_t>& is_driver_pin,
                            std::vector<uint8_t>& is_seq_output_pin);
void buildPowerNetDriverPins(GTDatabase& gtdb,
                             int n,
                             const std::vector<uint8_t>& is_driver_pin,
                             std::vector<int>& net_driver_pin);
void buildPowerClockGateMaps(GTDatabase& gtdb,
                             int n,
                             const std::vector<int>& pin_to_node,
                             std::vector<int>& clock_gate_out_for_input,
                             std::vector<int>& clock_gate_clock_for_out,
                             std::vector<int>& clock_gate_enable_for_out,
                             std::vector<uint8_t>& is_clock_gate_clock_pin);
struct PowerCpuActivityLevels {
    std::vector<int> pin_level;
    std::vector<int> level_list;
    std::vector<int> level_list_end;
    int max_level = 0;
};

PowerCpuActivityLevels buildPowerCpuActivityLevels(GTDatabase& gtdb, int n);
void buildPowerNodePortPinMap(GTDatabase& gtdb,
                              std::vector<int>& node_port_pin_start,
                              std::vector<int>& node_port_pin_list);

std::vector<int> buildPowerClockPins(GTDatabase& gtdb,
                                     int n,
                                     const std::vector<int>& pin_to_node,
                                     const std::vector<int>& pin_to_net,
                                     const std::vector<uint8_t>& is_load_pin,
                                     const std::vector<uint8_t>& is_driver_pin,
                                     const std::vector<uint8_t>& is_clock_gate_clock_pin);

}  // namespace gt
