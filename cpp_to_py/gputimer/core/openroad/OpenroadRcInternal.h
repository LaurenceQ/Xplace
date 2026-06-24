#pragma once

#include "gputimer/core/GPUTimer.h"
#include "common/utils/utils.h"
#include "common/StageProfiler.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Geometry.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "gputimer/db/GTDatabase.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gt {
namespace openroad_rc {

struct OpenroadRoutePt {
    int x = 0;
    int y = 0;
    int layer = 0;  // OpenROAD routing level, 1-based.
    bool valid = false;
};

struct OpenroadRoutePtKey {
    int x = 0;
    int y = 0;
    int layer = 0;

    bool operator==(const OpenroadRoutePtKey& rhs) const
    {
        return x == rhs.x && y == rhs.y && layer == rhs.layer;
    }
};

struct OpenroadRoutePtKeyHash {
    std::size_t operator()(const OpenroadRoutePtKey& key) const
    {
        std::size_t seed = static_cast<std::size_t>(key.x);
        seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.layer) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct OpenroadPinRouteLoc {
    int pin_x = 0;
    int pin_y = 0;
    int pin_layer = 1;
    int grid_src_x = 0;
    int grid_src_y = 0;
    int grid_x = 0;
    int grid_y = 0;
    int conn_layer = 1;
    bool valid = false;
};

struct RouteSegmentCacheMeta {
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
};

struct LocalRcNetGraph {
    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    std::vector<float> node_cap;
    std::vector<int> node2pin;
    std::vector<std::string> node_names;
    std::vector<OpenroadRoutePt> route_points;
    std::unordered_map<std::string, int> node_name2id;
    std::unordered_map<std::string, int> pin_name2id;
};

struct FlatLocalAdjacency {
    std::vector<int> start;
    std::vector<int> edge;
    std::vector<int> next;
};

using OpenroadRouteNodeMap =
    std::unordered_map<OpenroadRoutePtKey, int, OpenroadRoutePtKeyHash>;

struct OpenroadGrRcBuildCounts {
    int parsed_nets = 0;
    int missing_nets = 0;
    int unknown_nets = 0;
    int node_rows = 0;
    int pin_rows = 0;
    int cap_rows = 0;
    int resistors = 0;
    int skipped_self_resistors = 0;
    int unresolved_pin_nodes = 0;
    int missing_driver_nodes = 0;
    int missing_net_pins = 0;
    int fallback_net_pins = 0;
    int repaired_edges = 0;
    int skipped_loop_edges = 0;
    int malformed_rows = 0;
    int pin_net_mismatches = 0;
};

struct OpenroadRouteSegmentsBuildCounts {
    int parsed_nets = 0;
    int missing_nets = 0;
    int unknown_nets = 0;
    int segment_rows = 0;
    int wire_segments = 0;
    int via_segments = 0;
    int malformed_rows = 0;
    int unknown_layers = 0;
    int non_manhattan_segments = 0;
    int skipped_self_segments = 0;
    int missing_driver_nodes = 0;
    int missing_net_pins = 0;
    int fallback_net_pins = 0;
    int pin_stub_edges = 0;
    int skipped_missing_unconnected_nets = 0;
    long long skipped_missing_unconnected_pins = 0;
    int repaired_edges = 0;
    int skipped_loop_edges = 0;
    int skipped_missing_high_fanout_nets = 0;
    long long skipped_missing_high_fanout_pins = 0;

    void mergeParseCounts(const OpenroadRouteSegmentsBuildCounts& other)
    {
        segment_rows += other.segment_rows;
        wire_segments += other.wire_segments;
        via_segments += other.via_segments;
        malformed_rows += other.malformed_rows;
        unknown_layers += other.unknown_layers;
        non_manhattan_segments += other.non_manhattan_segments;
        skipped_self_segments += other.skipped_self_segments;
    }

    void mergeFinalizeCounts(const OpenroadRouteSegmentsBuildCounts& other)
    {
        missing_nets += other.missing_nets;
        missing_driver_nodes += other.missing_driver_nodes;
        missing_net_pins += other.missing_net_pins;
        fallback_net_pins += other.fallback_net_pins;
        pin_stub_edges += other.pin_stub_edges;
        skipped_missing_unconnected_nets += other.skipped_missing_unconnected_nets;
        skipped_missing_unconnected_pins += other.skipped_missing_unconnected_pins;
        skipped_missing_high_fanout_nets += other.skipped_missing_high_fanout_nets;
        skipped_missing_high_fanout_pins += other.skipped_missing_high_fanout_pins;
        repaired_edges += other.repaired_edges;
        skipped_loop_edges += other.skipped_loop_edges;
    }
};

struct OpenroadInferredGrid {
    int tile_size = 0;
    int origin_x = 0;
    int origin_y = 0;
    bool valid = false;
};

struct OpenroadRouteGridStats {
    bool have_x = false;
    bool have_y = false;
    int first_x = 0;
    int first_y = 0;
    int x_step = 0;
    int y_step = 0;
};

struct NangateLayerRc {
    float res_ohm_per_um = 0.0f;
    float cap_f_per_um = 0.0f;
};

struct OpenroadPinMapStats {
    int gpdb_direct_pins = 0;
    int unresolved_pins = 0;
    int name_resolved_pins = 0;
};

bool env_enabled(const char* name);
std::uint64_t route_segment_design_signature(const GTDatabase& gtdb,
                                             int expected_num_nets,
                                             int expected_num_pins);
RouteSegmentCacheMeta route_segment_cache_meta(const std::string& source_file);
std::string route_segment_cache_path(const std::string& source_file);
bool load_route_segment_cache(const std::string& cache_path,
                              const RouteSegmentCacheMeta& meta,
                              int expected_num_nets,
                              int expected_num_pins,
                              int missing_high_fanout_skip,
                              std::uint64_t expected_design_signature,
                              HostRcGraph& graph);
void save_route_segment_cache(const std::string& cache_path,
                              const RouteSegmentCacheMeta& meta,
                              int expected_num_nets,
                              int expected_num_pins,
                              int missing_high_fanout_skip,
                              std::uint64_t design_signature,
                              const HostRcGraph& graph);

std::string normalized_spef_name(std::string name);
bool spef_digits_only(const std::string& value);
void add_name_alias(std::unordered_map<std::string, int>& map,
                    const std::string& name,
                    int value);
std::string replace_char(std::string name, char from, char to);
std::string replace_last_char(std::string name, char from, char to);
void add_gr_name_alias(std::unordered_map<std::string, int>& map,
                       const std::string& name,
                       int value);
std::vector<std::string> split_tsv(const std::string& line);
bool parse_int_field(const std::string& value, int& out);
const char* skip_route_ws(const char* ptr, const char* end);
bool parse_int_token(const char*& ptr, const char* end, int& out);
bool parse_token_range(const char*& ptr,
                       const char* end,
                       const char*& begin,
                       const char*& finish);
bool route_rest_is_ws(const char* ptr, const char* end);
bool parse_route_segment_row(const std::string& line,
                             int& x1,
                             int& y1,
                             const char*& layer1_begin,
                             const char*& layer1_end,
                             int& x2,
                             int& y2,
                             const char*& layer2_begin,
                             const char*& layer2_end);
bool parse_float_field(const std::string& value, float& out);
std::vector<std::string> split_whitespace(const std::string& line);
std::string lowercase_string(std::string value);
int trailing_integer(const std::string& value);
int trailing_integer_token(const char* begin, const char* end);
int resolve_route_net_token(const std::unordered_map<std::string_view, int>& net_name_to_index,
                            const char* begin,
                            const char* end);
int resolve_route_layer_token(const std::unordered_map<std::string, int>& layer_name_to_level,
                              const char* begin,
                              const char* end);

void add_attr_cap(std::vector<float>& node_cap, int node, float cap);
void set_attr_cap(std::vector<float>& node_cap, int node, float cap);
void set_attr_cap(std::vector<float>& node_cap, int node, int attr, float cap);
float pin_cap_attr_host(const GTDatabase& gtdb, int pin, int attr);
std::string openroad_gr_edge_key(int from, int to, const std::string& res_id);
void ensure_local_node(LocalRcNetGraph& local, int node_id);
std::string spef_upper(std::string value);
bool spef_includes_pin_caps_from_design_flow(const std::string& design_flow);
int count_tree_edges_from_root(const LocalRcNetGraph& local);
FlatLocalAdjacency build_flat_local_adjacency(const LocalRcNetGraph& local);
int prune_to_rooted_tree(LocalRcNetGraph& local);
int append_local_rc_node(LocalRcNetGraph& local,
                         int pin_id,
                         const std::string& name,
                         const OpenroadRoutePt& route_pt,
                         bool keep_route_node_names);
int append_pin_node(LocalRcNetGraph& local,
                    int pin_id,
                    const std::vector<std::string>& pin_names,
                    bool keep_route_node_names);
int append_route_node(int net_idx,
                      const OpenroadRoutePtKey& key,
                      std::vector<std::unique_ptr<LocalRcNetGraph>>& local_nets,
                      std::vector<std::unique_ptr<OpenroadRouteNodeMap>>& route_node_maps,
                      bool keep_route_node_names);
void add_edge(LocalRcNetGraph& local, int from, int to, float res);
void reorder_root(LocalRcNetGraph& local, int root_node);

bool route_point_matches(const OpenroadRoutePt& pt, const OpenroadRoutePtKey& key);
int openroad_layer_track_spacing(const db::Layer& layer);
int openroad_gcell_tile_size(const db::Database& rawdb);
std::pair<int, int> openroad_position_on_grid(const db::Database& rawdb,
                                              int tile_size,
                                              int x,
                                              int y);
int positive_mod(int value, int modulus);
int infer_openroad_grid_step(std::vector<int> coords);
int infer_openroad_grid_origin(const std::vector<int>& coords,
                               int tile_size,
                               int fallback_origin);
int infer_openroad_grid_origin_from_first(int first_coord,
                                          bool have_coord,
                                          int tile_size,
                                          int fallback_origin);
void add_openroad_route_grid_point(OpenroadRouteGridStats& stats, int x, int y);
OpenroadInferredGrid infer_openroad_route_grid_from_stats(
    const std::vector<OpenroadRouteGridStats>& thread_stats,
    const db::Database& rawdb,
    int fallback_tile_size);
OpenroadInferredGrid infer_openroad_route_grid(
    const std::vector<std::unique_ptr<LocalRcNetGraph>>& local_nets,
    const db::Database& rawdb,
    int fallback_tile_size,
    int threads);
std::pair<int, int> openroad_position_on_inferred_grid(
    const db::Database& rawdb,
    const OpenroadInferredGrid& grid,
    int x,
    int y);
NangateLayerRc nangate45_layer_rc(int routing_level);
float nangate45_via_res_ohm(int lower_routing_level);
std::tuple<int, int, int, int> orient_box_for_iopin(int orient,
                                                    int lx,
                                                    int ly,
                                                    int hx,
                                                    int hy);
std::tuple<int, int, int, int> orient_box_for_cell(const db::CellType* ctype,
                                                   int orient,
                                                   int lx,
                                                   int ly,
                                                   int hx,
                                                   int hy);
OpenroadPinMapStats resolve_openroad_timer_pins(const GTDatabase& gtdb,
                                                int num_pins,
                                                std::vector<db::Pin*>& pin_id_to_dbpin,
                                                int threads);
bool openroad_pin_route_loc(const GTDatabase& gtdb,
                            const std::vector<db::Pin*>& pin_id_to_dbpin,
                            const std::vector<const db::Layer*>& routing_level_to_layer,
                            const OpenroadInferredGrid& openroad_grid,
                            int pin_id,
                            OpenroadPinRouteLoc& loc);

}  // namespace openroad_rc
}  // namespace gt
