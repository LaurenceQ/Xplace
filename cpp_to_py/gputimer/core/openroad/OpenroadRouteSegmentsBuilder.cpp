#include "gputimer/core/openroad/OpenroadRcInternal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gt {

using namespace openroad_rc;

namespace {

struct RouteSegmentMappedFile {
    int fd = -1;
    const char* data = nullptr;
    std::size_t size = 0;

    explicit RouteSegmentMappedFile(const std::string& file)
    {
        fd = ::open(file.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Cannot open OpenROAD route segment file: " + file +
                                     ": " + std::strerror(errno));
        }

        struct stat st;
        if (::fstat(fd, &st) != 0) {
            const std::string error = std::strerror(errno);
            ::close(fd);
            fd = -1;
            throw std::runtime_error("Cannot stat OpenROAD route segment file: " + file +
                                     ": " + error);
        }
        if (st.st_size <= 0) {
            size = 0;
            return;
        }
        size = static_cast<std::size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            const std::string error = std::strerror(errno);
            ::close(fd);
            fd = -1;
            throw std::runtime_error("Cannot mmap OpenROAD route segment file: " + file +
                                     ": " + error);
        }
        data = static_cast<const char*>(mapped);
    }

    RouteSegmentMappedFile(const RouteSegmentMappedFile&) = delete;
    RouteSegmentMappedFile& operator=(const RouteSegmentMappedFile&) = delete;

    ~RouteSegmentMappedFile()
    {
        if (data != nullptr) {
            ::munmap(const_cast<char*>(data), size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

struct RouteSegmentBlock {
    int net_idx = -1;
    std::size_t begin = 0;
    std::size_t end = 0;
};

bool parse_route_segment_row_range(const char* line_begin,
                                   const char* line_end,
                                   int& x1,
                                   int& y1,
                                   const char*& layer1_begin,
                                   const char*& layer1_end,
                                   int& x2,
                                   int& y2,
                                   const char*& layer2_begin,
                                   const char*& layer2_end)
{
    const char* ptr = line_begin;
    if (!parse_int_token(ptr, line_end, x1) ||
        !parse_int_token(ptr, line_end, y1) ||
        !parse_token_range(ptr, line_end, layer1_begin, layer1_end) ||
        !parse_int_token(ptr, line_end, x2) ||
        !parse_int_token(ptr, line_end, y2) ||
        !parse_token_range(ptr, line_end, layer2_begin, layer2_end)) {
        return false;
    }
    return route_rest_is_ws(ptr, line_end);
}

void add_route_segment_parse_stats(OpenroadRouteSegmentsBuildStats& dst,
                                   const OpenroadRouteSegmentsBuildStats& src)
{
    dst.segment_rows += src.segment_rows;
    dst.wire_segments += src.wire_segments;
    dst.via_segments += src.via_segments;
    dst.malformed_rows += src.malformed_rows;
    dst.unknown_layers += src.unknown_layers;
    dst.non_manhattan_segments += src.non_manhattan_segments;
    dst.skipped_self_segments += src.skipped_self_segments;
}

}  // namespace

HostRcGraph GPUTimer::build_openroad_route_segments_rc(const std::string& file) {

    const bool profile = env_enabled("GPUTIMER_ROUTE_SEG_PROFILE");
    const auto build_start = std::chrono::steady_clock::now();
    auto seconds_since = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    auto profile_log = [&](const char* phase) {
        if (profile) {
            std::fprintf(stderr,
                         "[ROUTE_SEG_PROFILE] phase=%s elapsed=%.3f\n",
                         phase,
                         seconds_since(build_start));
            std::fflush(stderr);
        }
    };

    const char* debug_pin_net_env = std::getenv("GPUTIMER_DEBUG_ROUTE_PIN_NET");
    const std::string debug_pin_net =
        debug_pin_net_env == nullptr ? std::string() : normalized_spef_name(debug_pin_net_env);
    int missing_high_fanout_skip = 300;
    if (const char* skip_env = std::getenv("GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP")) {
        missing_high_fanout_skip = std::atoi(skip_env);
        if (missing_high_fanout_skip < 0) {
            missing_high_fanout_skip = 0;
        }
    }
    const bool keep_route_node_names = env_enabled("GPUTIMER_ROUTE_SEG_KEEP_NODE_NAMES");
    const bool cache_enabled =
        !profile &&
        debug_pin_net.empty() &&
        !keep_route_node_names &&
        !env_enabled("GPUTIMER_ROUTE_SEG_DISABLE_CACHE");
    const bool cache_profile = env_enabled("GPUTIMER_ROUTE_SEG_CACHE_PROFILE") ||
                               env_enabled("DMP_RC_PROFILE");
    const auto cache_start = std::chrono::steady_clock::now();
    auto cache_last = cache_start;
    auto cache_profile_log = [&](const char* phase) {
        if (!cache_profile) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - cache_last).count();
        const double total = std::chrono::duration<double>(now - cache_start).count();
        std::fprintf(stderr,
                     "[ROUTE_SEG_CACHE_PROFILE] phase=%s elapsed=%.3f total=%.3f cache_enabled=%d\n",
                     phase,
                     elapsed,
                     total,
                     cache_enabled ? 1 : 0);
        std::fflush(stderr);
        cache_last = now;
    };
    RouteSegmentCacheMeta cache_meta;
    std::string cache_path;
    std::uint64_t cache_design_signature = 0;
    if (cache_enabled) {
        cache_meta = route_segment_cache_meta(file);
        cache_profile_log("cache_meta");
        cache_path = route_segment_cache_path(file);
        cache_profile_log("cache_path");
        cache_design_signature = route_segment_design_signature(gtdb, num_nets, num_pins);
        cache_profile_log("design_signature");
    } else {
        cache_profile_log("cache_disabled");
    }
    if (cache_enabled && cache_meta.source_size > 0) {
        HostRcGraph cached_graph;
        if (load_route_segment_cache(cache_path, cache_meta, num_nets, num_pins,
                                     missing_high_fanout_skip,
                                     cache_design_signature,
                                     cached_graph)) {
            cache_profile_log("cache_hit_load_return");
            logger.info("OpenROAD route-segment RC graph loaded from cache: %s",
                        cache_path.c_str());
            return cached_graph;
        }
        cache_profile_log("cache_miss_or_invalid");
    }

    RouteSegmentMappedFile mapped_file(file);
    profile_log("open_file");

    torch::Tensor flat_net2pin_start_map_at =
        timing_raw_db.flat_net2pin_start_map.cpu().contiguous();
    torch::Tensor flat_net2pin_map_at =
        timing_raw_db.flat_net2pin_map.cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();
    profile_log("copy_net_pin_map_to_host");

    const float dbu_per_micron = static_cast<float>(gtdb.rawdb.DBU_Micron);
    if (!(dbu_per_micron > 0.0f)) {
        throw std::runtime_error("OpenROAD route segment RC requires a positive DBU/micron value.");
    }

    std::unordered_map<std::string_view, int> net_name_to_index;
    net_name_to_index.reserve(gtdb.net_names.size() * 2);
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        if (!gtdb.net_names[i].empty()) {
            net_name_to_index.emplace(std::string_view(gtdb.net_names[i]), i);
        }
    }
    profile_log("build_net_name_map");

    std::unordered_map<std::string, int> layer_name_to_level;
    for (const db::Layer& layer : gtdb.rawdb.layers) {
        if (layer.rIndex >= 0) {
            layer_name_to_level.emplace(lowercase_string(layer.name()), layer.rIndex + 1);
        }
    }

    std::vector<db::Pin*> pin_id_to_dbpin(num_pins, nullptr);
    const OpenroadPinMapStats pin_map_stats =
        resolve_openroad_timer_pins(gtdb, num_pins, pin_id_to_dbpin);
    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=map_pins_by_gpdb elapsed=%.3f gpdb_direct=%d unresolved=%d total_timer_pins=%d\n",
                     seconds_since(build_start),
                     pin_map_stats.gpdb_direct_pins,
                     pin_map_stats.unresolved_pins,
                     num_pins);
        std::fflush(stderr);
    }
    profile_log(pin_map_stats.unresolved_pins > 0
                    ? "build_dbpin_alias_map"
                    : "build_dbpin_alias_map_skipped");
    logger.info("OpenROAD route-segment pin mapping: gpdb_direct=%d name_resolved=%d total_timer_pins=%d",
                pin_map_stats.gpdb_direct_pins, pin_map_stats.name_resolved_pins, num_pins);
    profile_log("resolve_timer_pins");

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=route_segment_options elapsed=%.3f missing_high_fanout_skip=%d\n",
                     seconds_since(build_start),
                     missing_high_fanout_skip);
        std::fflush(stderr);
    }

    OpenroadRouteSegmentsBuildStats stats;
    std::vector<std::unique_ptr<LocalSpefNetRc>> local_nets(num_nets);
    std::vector<std::unique_ptr<OpenroadRouteNodeMap>> route_node_maps(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);

    std::vector<RouteSegmentBlock> route_blocks;
    route_blocks.reserve(static_cast<std::size_t>(num_nets));
    long long raw_lines = 0;
    bool duplicate_route_blocks = false;
    int current_net = -1;
    const char* route_data = mapped_file.data;
    const std::size_t route_size = mapped_file.size;

    auto close_current_block = [&](std::size_t end_offset) {
        if (current_net >= 0 && !route_blocks.empty()) {
            route_blocks.back().end = end_offset;
        }
        current_net = -1;
    };

    for (std::size_t line_begin_offset = 0; line_begin_offset < route_size;) {
        std::size_t line_end_offset = line_begin_offset;
        while (line_end_offset < route_size && route_data[line_end_offset] != '\n') {
            ++line_end_offset;
        }
        const std::size_t next_line_offset =
            line_end_offset < route_size ? line_end_offset + 1 : line_end_offset;
        ++raw_lines;

        const char* line_begin = route_data + line_begin_offset;
        const char* line_end = route_data + line_end_offset;
        const char* first = skip_route_ws(line_begin, line_end);
        if (first == line_end || *first == '#') {
            line_begin_offset = next_line_offset;
            continue;
        }

        const char* token_end = first;
        while (token_end < line_end &&
               !std::isspace(static_cast<unsigned char>(*token_end))) {
            ++token_end;
        }
        const bool one_token = route_rest_is_ws(token_end, line_end);
        if (one_token) {
            if ((token_end - first) == 1 && *first == '(') {
                line_begin_offset = next_line_offset;
                continue;
            }
            if ((token_end - first) == 1 && *first == ')') {
                close_current_block(line_begin_offset);
                line_begin_offset = next_line_offset;
                continue;
            }

            close_current_block(line_begin_offset);
            const int net_idx = resolve_route_net_token(net_name_to_index, first, token_end);
            if (net_idx < 0 || net_idx >= num_nets) {
                stats.unknown_nets++;
                line_begin_offset = next_line_offset;
                continue;
            }
            if (parsed_net[net_idx]) {
                duplicate_route_blocks = true;
            } else {
                parsed_net[net_idx] = 1;
                stats.parsed_nets++;
            }
            current_net = net_idx;
            route_blocks.push_back(RouteSegmentBlock{net_idx, next_line_offset, route_size});
            line_begin_offset = next_line_offset;
            continue;
        }

        if (current_net < 0) {
            stats.malformed_rows++;
        }
        line_begin_offset = next_line_offset;
    }
    close_current_block(route_size);

    int parse_threads = std::max(1, num_threads);
    if (const char* thread_env = std::getenv("GPUTIMER_ROUTE_SEG_PARSE_THREADS")) {
        const int env_threads = std::atoi(thread_env);
        if (env_threads > 0) {
            parse_threads = env_threads;
        }
    }
    if (!debug_pin_net.empty() || duplicate_route_blocks) {
        parse_threads = 1;
    }

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=scan_route_blocks elapsed=%.3f raw_lines=%lld blocks=%zu parsed_nets=%d unknown_nets=%d duplicate_blocks=%d parse_threads=%d\n",
                     seconds_since(build_start),
                     raw_lines,
                     route_blocks.size(),
                     stats.parsed_nets,
                     stats.unknown_nets,
                     duplicate_route_blocks ? 1 : 0,
                     parse_threads);
        std::fflush(stderr);
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=parse_segments_parallel_start elapsed=%.3f threads=%d blocks=%zu\n",
                     seconds_since(build_start),
                     parse_threads,
                     route_blocks.size());
        std::fflush(stderr);
    }

    std::vector<OpenroadRouteSegmentsBuildStats> parse_stats(parse_threads);
    const auto parse_start = std::chrono::steady_clock::now();

    auto parse_one_block = [&](const RouteSegmentBlock& block,
                               OpenroadRouteSegmentsBuildStats& local_stats) {
        for (std::size_t line_begin_offset = block.begin;
             line_begin_offset < block.end;) {
            std::size_t line_end_offset = line_begin_offset;
            while (line_end_offset < block.end && route_data[line_end_offset] != '\n') {
                ++line_end_offset;
            }
            const std::size_t next_line_offset =
                line_end_offset < block.end ? line_end_offset + 1 : line_end_offset;

            const char* line_begin = route_data + line_begin_offset;
            const char* line_end = route_data + line_end_offset;
            const char* first = skip_route_ws(line_begin, line_end);
            if (first == line_end || *first == '#') {
                line_begin_offset = next_line_offset;
                continue;
            }
            const char* token_end = first;
            while (token_end < line_end &&
                   !std::isspace(static_cast<unsigned char>(*token_end))) {
                ++token_end;
            }
            if (route_rest_is_ws(token_end, line_end) &&
                (token_end - first) == 1 &&
                (*first == '(' || *first == ')')) {
                line_begin_offset = next_line_offset;
                continue;
            }

            int x1 = 0;
            int y1 = 0;
            int x2 = 0;
            int y2 = 0;
            const char* layer1_begin = nullptr;
            const char* layer1_end = nullptr;
            const char* layer2_begin = nullptr;
            const char* layer2_end = nullptr;
            if (!parse_route_segment_row_range(line_begin, line_end, x1, y1,
                                               layer1_begin, layer1_end,
                                               x2, y2,
                                               layer2_begin, layer2_end)) {
                local_stats.malformed_rows++;
                line_begin_offset = next_line_offset;
                continue;
            }
            const int layer1 = resolve_route_layer_token(layer_name_to_level,
                                                         layer1_begin,
                                                         layer1_end);
            const int layer2 = resolve_route_layer_token(layer_name_to_level,
                                                         layer2_begin,
                                                         layer2_end);
            if (layer1 <= 0 || layer2 <= 0) {
                local_stats.unknown_layers++;
                line_begin_offset = next_line_offset;
                continue;
            }

            const bool is_manhattan = (x1 == x2) || (y1 == y2);
            if (!is_manhattan) {
                local_stats.non_manhattan_segments++;
                line_begin_offset = next_line_offset;
                continue;
            }

            const OpenroadRoutePtKey key1{x1, y1, layer1};
            const OpenroadRoutePtKey key2{x2, y2, layer2};
            const int from = append_route_node(block.net_idx, key1, local_nets,
                                               route_node_maps, keep_route_node_names);
            const int to = append_route_node(block.net_idx, key2, local_nets,
                                             route_node_maps, keep_route_node_names);
            if (from == to) {
                local_stats.skipped_self_segments++;
                line_begin_offset = next_line_offset;
                continue;
            }

            long long dx = static_cast<long long>(x1) - static_cast<long long>(x2);
            long long dy = static_cast<long long>(y1) - static_cast<long long>(y2);
            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            const long long length_dbu = dx + dy;
            LocalSpefNetRc& local = *local_nets[block.net_idx];
            if (length_dbu == 0) {
                const int lower_layer = std::min(layer1, layer2);
                add_edge(local, from, to,
                         nangate45_via_res_ohm(lower_layer) / gtdb.res_unit);
                local_stats.via_segments++;
            } else if (layer1 == layer2) {
                const NangateLayerRc rc = nangate45_layer_rc(layer1);
                if (!(rc.res_ohm_per_um > 0.0f) && !(rc.cap_f_per_um > 0.0f)) {
                    local_stats.unknown_layers++;
                    line_begin_offset = next_line_offset;
                    continue;
                }
                const float length_um = static_cast<float>(length_dbu) / dbu_per_micron;
                const float edge_res = (rc.res_ohm_per_um * length_um) / gtdb.res_unit;
                const float cap = (rc.cap_f_per_um * length_um) / gtdb.cap_unit;
                add_edge(local, from, to, edge_res);
                add_attr_cap(local.node_cap, from, cap * 0.5f);
                add_attr_cap(local.node_cap, to, cap * 0.5f);
                local_stats.wire_segments++;
            } else {
                local_stats.non_manhattan_segments++;
                line_begin_offset = next_line_offset;
                continue;
            }
            local_stats.segment_rows++;
            line_begin_offset = next_line_offset;
        }
    };

#pragma omp parallel for num_threads(parse_threads) schedule(dynamic, 256)
    for (int block_idx = 0; block_idx < static_cast<int>(route_blocks.size()); ++block_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        parse_one_block(route_blocks[block_idx], parse_stats[tid]);
    }

    double parse_wall_seconds = seconds_since(parse_start);
    for (const OpenroadRouteSegmentsBuildStats& thread_stats : parse_stats) {
        add_route_segment_parse_stats(stats, thread_stats);
    }
    route_blocks.clear();
    route_blocks.shrink_to_fit();

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=parse_segments elapsed=%.3f wall=%.3f raw_lines=%lld segment_rows=%d parsed_nets=%d route_nodes_pending=%zu\n",
                     seconds_since(build_start),
                     parse_wall_seconds,
                     raw_lines,
                     stats.segment_rows,
                     stats.parsed_nets,
                     static_cast<std::size_t>(stats.parsed_nets));
        std::fflush(stderr);
    }

    const int openroad_tile_size = openroad_gcell_tile_size(gtdb.rawdb);
    if (openroad_tile_size <= 0) {
        throw std::runtime_error("OpenROAD route segment RC requires a positive OpenROAD gcell tile size.");
    }
    const OpenroadInferredGrid openroad_grid =
        infer_openroad_route_grid(local_nets, gtdb.rawdb, openroad_tile_size);
    if (!openroad_grid.valid) {
        throw std::runtime_error("OpenROAD route segment RC could not infer the route segment grid.");
    }
    profile_log("infer_route_grid");
    if (!debug_pin_net.empty()) {
        std::fprintf(stderr,
                     "[ROUTE GRID DEBUG] fallback_tile=%d inferred_tile=%d origin=(%d,%d)\n",
                     openroad_tile_size,
                     openroad_grid.tile_size,
                     openroad_grid.origin_x,
                     openroad_grid.origin_y);
    }

    HostRcGraph graph;
    graph.includes_pin_caps.assign(num_nets, 0);

    struct RouteFinalizeThreadStats {
        OpenroadRouteSegmentsBuildStats stats;
        double pinloc_seconds = 0.0;
        double attach_seconds = 0.0;
        double reorder_seconds = 0.0;
        double repair_seconds = 0.0;
        double prune_seconds = 0.0;
        int repair_adjacency_nets = 0;
        int repair_scan_nets = 0;
        long long repair_node_edge_product_max = 0;
    };

    int finalize_threads = std::max(1, num_threads);
    if (const char* thread_env = std::getenv("GPUTIMER_ROUTE_SEG_FINALIZE_THREADS")) {
        const int env_threads = std::atoi(thread_env);
        if (env_threads > 0) {
            finalize_threads = env_threads;
        }
    }
    if (!debug_pin_net.empty()) {
        finalize_threads = 1;
    }

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=finalize_parallel_start elapsed=%.3f threads=%d\n",
                     seconds_since(build_start),
                     finalize_threads);
        std::fflush(stderr);
    }

    std::vector<int> net_node_count(num_nets, 0);
    std::vector<int> net_edge_count(num_nets, 0);
    std::vector<RouteFinalizeThreadStats> finalize_stats(finalize_threads);
    const auto finalize_start = std::chrono::steady_clock::now();

    auto finalize_one_net = [&](int net_idx, RouteFinalizeThreadStats& thread_stats) {
        auto phase_start = std::chrono::steady_clock::now();
        OpenroadRouteSegmentsBuildStats& local_stats = thread_stats.stats;
        const int pin_begin = flat_net2pin_start_map[net_idx];
        const int pin_end = flat_net2pin_start_map[net_idx + 1];
        const int fanout = pin_end - pin_begin;
        const int driver_pin = pin_begin < pin_end ? flat_net2pin_map[pin_begin] : -1;
        const bool net_parsed = parsed_net[net_idx] != 0;

        if (!net_parsed) {
            local_stats.missing_nets++;
        }

        if (!net_parsed && fanout <= 1) {
            local_stats.skipped_missing_unconnected_nets++;
            local_stats.skipped_missing_unconnected_pins += std::max(fanout, 0);
            local_nets[net_idx].reset();
            return;
        }

        auto ensure_local = [&]() -> LocalSpefNetRc& {
            if (!local_nets[net_idx]) {
                local_nets[net_idx] = std::make_unique<LocalSpefNetRc>();
            }
            return *local_nets[net_idx];
        };
        LocalSpefNetRc& local = ensure_local();

        if (!net_parsed && missing_high_fanout_skip > 0 && fanout > missing_high_fanout_skip) {
            local_stats.skipped_missing_high_fanout_nets++;
            local_stats.skipped_missing_high_fanout_pins += fanout;
            if (driver_pin >= 0) {
                const int driver_node = append_pin_node(local, driver_pin, gtdb.pin_names, keep_route_node_names);
                for (int pin_pos = pin_begin; pin_pos < pin_end; ++pin_pos) {
                    const int pin_id = flat_net2pin_map[pin_pos];
                    if (pin_id < 0 || pin_id == driver_pin) {
                        continue;
                    }
                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                        const float cap = pin_cap_attr_host(gtdb, pin_id, attr);
                        if (cap > 0.0f) {
                            local.node_cap[driver_node * NUM_ATTR + attr] += cap;
                        }
                    }
                }
            }
            net_node_count[net_idx] = static_cast<int>(local.node2pin.size());
            net_edge_count[net_idx] = static_cast<int>(local.edge_from.size());
            return;
        }

        int min_route_layer = std::numeric_limits<int>::max();
        int max_route_layer = 0;
        for (const OpenroadRoutePt& pt : local.route_points) {
            if (!pt.valid) {
                continue;
            }
            min_route_layer = std::min(min_route_layer, pt.layer);
            max_route_layer = std::max(max_route_layer, pt.layer);
        }
        if (min_route_layer == std::numeric_limits<int>::max()) {
            min_route_layer = 0;
        }

        std::vector<OpenroadPinRouteLoc> pin_route_locs(pin_end - pin_begin);
        for (int pin_pos = pin_begin; pin_pos < pin_end; ++pin_pos) {
            const int pin_id = flat_net2pin_map[pin_pos];
            OpenroadPinRouteLoc loc;
            openroad_pin_route_loc(gtdb, pin_id_to_dbpin, openroad_grid, pin_id, loc);
            pin_route_locs[pin_pos - pin_begin] = loc;
            if (!debug_pin_net.empty() &&
                net_idx < static_cast<int>(gtdb.net_names.size()) &&
                normalized_spef_name(gtdb.net_names[net_idx]) == debug_pin_net) {
                db::Pin* dbpin = pin_id >= 0 &&
                                 pin_id < static_cast<int>(pin_id_to_dbpin.size())
                                     ? pin_id_to_dbpin[pin_id]
                                     : nullptr;
                std::string dbpin_name = "<null>";
                if (dbpin != nullptr && dbpin->cell != nullptr && dbpin->type != nullptr) {
                    dbpin_name = dbpin->cell->name() + ":" + dbpin->type->name();
                } else if (dbpin != nullptr && dbpin->iopin != nullptr) {
                    dbpin_name = dbpin->iopin->name;
                }
                std::fprintf(stderr,
                             "[ROUTE PIN DEBUG] net=%s pin_id=%d timer_pin=%s dbpin=%s valid=%d pin=(%d,%d,L%d) grid_src=(%d,%d) grid=(%d,%d,L%d)\n",
                             gtdb.net_names[net_idx].c_str(),
                             pin_id,
                             (pin_id >= 0 && pin_id < static_cast<int>(gtdb.pin_names.size()))
                                 ? gtdb.pin_names[pin_id].c_str()
                                 : "<bad>",
                             dbpin_name.c_str(),
                             loc.valid ? 1 : 0,
                             loc.pin_x,
                             loc.pin_y,
                             loc.pin_layer,
                             loc.grid_src_x,
                             loc.grid_src_y,
                             loc.grid_x,
                             loc.grid_y,
                             loc.conn_layer);
            }
        }
        thread_stats.pinloc_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        if (local.node2pin.empty() && driver_pin >= 0) {
            append_pin_node(local, driver_pin, gtdb.pin_names, keep_route_node_names);
        }

        std::unordered_set<int> present_pins;
        present_pins.reserve(local.node2pin.size() + std::max(fanout, 0));
        for (int pin : local.node2pin) {
            if (pin >= 0) {
                present_pins.insert(pin);
            }
        }

        for (int pin_pos = pin_begin; pin_pos < pin_end; ++pin_pos) {
            const int pin_id = flat_net2pin_map[pin_pos];
            if (!present_pins.insert(pin_id).second) {
                continue;
            }

            const int pin_node = append_pin_node(local, pin_id, gtdb.pin_names, keep_route_node_names);
            const OpenroadPinRouteLoc& pin_loc = pin_route_locs[pin_pos - pin_begin];
            const bool have_pin_loc = pin_loc.valid;

            int route_node = -1;
            if (have_pin_loc && min_route_layer > 0) {
                OpenroadRouteNodeMap* route_map = route_node_maps[net_idx].get();
                auto find_grid_route_node = [&](int layer) {
                    const OpenroadRoutePtKey key{pin_loc.grid_x, pin_loc.grid_y, layer};
                    if (route_map == nullptr) {
                        for (int node = 0; node < static_cast<int>(local.route_points.size()); ++node) {
                            if (route_point_matches(local.route_points[node], key)) {
                                return node;
                            }
                        }
                        return -1;
                    }
                    auto iter = route_map->find(key);
                    return iter == route_map->end() ? -1 : iter->second;
                };
                if (pin_loc.conn_layer + 1 <= max_route_layer) {
                    route_node = find_grid_route_node(pin_loc.conn_layer + 1);
                }
                if (route_node < 0) {
                    route_node = find_grid_route_node(pin_loc.conn_layer);
                }
            }
            if (route_node < 0 && pin_node > 0) {
                if (net_parsed) {
                    local_stats.missing_net_pins++;
                } else {
                    local_stats.fallback_net_pins++;
                }
            }

            if (route_node >= 0 && route_node != pin_node) {
                float edge_res = 0.0f;
                if (have_pin_loc &&
                    route_node < static_cast<int>(local.route_points.size()) &&
                    local.route_points[route_node].valid) {
                    const OpenroadRoutePt& route_pt = local.route_points[route_node];
                    long long dx = static_cast<long long>(pin_loc.grid_x) - static_cast<long long>(pin_loc.pin_x);
                    long long dy = static_cast<long long>(pin_loc.grid_y) - static_cast<long long>(pin_loc.pin_y);
                    if (dx < 0) dx = -dx;
                    if (dy < 0) dy = -dy;
                    const float length_um = static_cast<float>(dx + dy) / dbu_per_micron;
                    const NangateLayerRc rc = nangate45_layer_rc(route_pt.layer);
                    edge_res = (rc.res_ohm_per_um * length_um) / gtdb.res_unit;
                    if (route_pt.layer == pin_loc.conn_layer + 1) {
                        edge_res += nangate45_via_res_ohm(route_pt.layer - 1) / gtdb.res_unit;
                    }
                    const float cap = (rc.cap_f_per_um * length_um) / gtdb.cap_unit;
                    add_attr_cap(local.node_cap, pin_node, cap * 0.5f);
                    add_attr_cap(local.node_cap, route_node, cap * 0.5f);
                    local_stats.pin_stub_edges++;
                } else if (net_parsed) {
                    local_stats.missing_net_pins++;
                } else {
                    local_stats.fallback_net_pins++;
                }
                add_edge(local, pin_node, route_node, edge_res);
            }
        }
        thread_stats.attach_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        int driver_node = -1;
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            if (local.node2pin[node] == driver_pin) {
                driver_node = node;
                break;
            }
        }
        if (driver_pin >= 0 && driver_node < 0) {
            driver_node = append_pin_node(local, driver_pin, gtdb.pin_names, keep_route_node_names);
            local_stats.missing_driver_nodes++;
        }
        reorder_root(local, driver_node);
        thread_stats.reorder_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);

            const long long node_edge_product =
                static_cast<long long>(local.node2pin.size()) *
                static_cast<long long>(local.edge_from.size());
            thread_stats.repair_node_edge_product_max =
                std::max(thread_stats.repair_node_edge_product_max, node_edge_product);

            if (node_edge_product > 4096) {
                ++thread_stats.repair_adjacency_nets;
                std::vector<std::vector<int>> adjacency(local.node2pin.size());
                for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                    const int from = local.edge_from[edge];
                    const int to = local.edge_to[edge];
                    if (from >= 0 && to >= 0 &&
                        from < static_cast<int>(adjacency.size()) &&
                        to < static_cast<int>(adjacency.size())) {
                        adjacency[from].emplace_back(to);
                        adjacency[to].emplace_back(from);
                    }
                }
                for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
                    const int node = stack[cursor];
                    for (int next : adjacency[node]) {
                        if (!seen[next]) {
                            seen[next] = 1;
                            stack.emplace_back(next);
                        }
                    }
                }
            } else {
                ++thread_stats.repair_scan_nets;
                for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
                    const int node = stack[cursor];
                    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                        int next = -1;
                        if (local.edge_from[edge] == node) next = local.edge_to[edge];
                        if (local.edge_to[edge] == node) next = local.edge_from[edge];
                        if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                            seen[next] = 1;
                            stack.emplace_back(next);
                        }
                    }
                }
            }
            for (int node = 1; node < static_cast<int>(seen.size()); ++node) {
                if (!seen[node]) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    local_stats.repaired_edges++;
                }
            }
        }
        thread_stats.repair_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        const int skipped_loop_edges = prune_to_rooted_tree(local);
        if (skipped_loop_edges > 0) {
            local_stats.skipped_loop_edges += skipped_loop_edges;
        }
        thread_stats.prune_seconds += seconds_since(phase_start);

        net_node_count[net_idx] = static_cast<int>(local.node2pin.size());
        net_edge_count[net_idx] = static_cast<int>(local.edge_from.size());
    };

#pragma omp parallel for num_threads(finalize_threads) schedule(dynamic, 1024)
    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        finalize_one_net(net_idx, finalize_stats[tid]);
    }

    double final_pinloc_seconds = 0.0;
    double final_attach_seconds = 0.0;
    double final_reorder_seconds = 0.0;
    double final_repair_seconds = 0.0;
    double final_prune_seconds = 0.0;
    double final_append_seconds = 0.0;
    int repair_adjacency_nets = 0;
    int repair_scan_nets = 0;
    long long repair_node_edge_product_max = 0;
    for (const RouteFinalizeThreadStats& thread_stats : finalize_stats) {
        const OpenroadRouteSegmentsBuildStats& local_stats = thread_stats.stats;
        stats.missing_nets += local_stats.missing_nets;
        stats.missing_driver_nodes += local_stats.missing_driver_nodes;
        stats.missing_net_pins += local_stats.missing_net_pins;
        stats.fallback_net_pins += local_stats.fallback_net_pins;
        stats.pin_stub_edges += local_stats.pin_stub_edges;
        stats.skipped_missing_unconnected_nets += local_stats.skipped_missing_unconnected_nets;
        stats.skipped_missing_unconnected_pins += local_stats.skipped_missing_unconnected_pins;
        stats.skipped_missing_high_fanout_nets += local_stats.skipped_missing_high_fanout_nets;
        stats.skipped_missing_high_fanout_pins += local_stats.skipped_missing_high_fanout_pins;
        stats.repaired_edges += local_stats.repaired_edges;
        stats.skipped_loop_edges += local_stats.skipped_loop_edges;
        final_pinloc_seconds += thread_stats.pinloc_seconds;
        final_attach_seconds += thread_stats.attach_seconds;
        final_reorder_seconds += thread_stats.reorder_seconds;
        final_repair_seconds += thread_stats.repair_seconds;
        final_prune_seconds += thread_stats.prune_seconds;
        repair_adjacency_nets += thread_stats.repair_adjacency_nets;
        repair_scan_nets += thread_stats.repair_scan_nets;
        repair_node_edge_product_max = std::max(repair_node_edge_product_max,
                                                thread_stats.repair_node_edge_product_max);
    }

    const double finalize_wall_seconds = seconds_since(finalize_start);
    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=finalize_parallel_done elapsed=%.3f wall=%.3f pinloc_work=%.3f attach_work=%.3f reorder_work=%.3f repair_work=%.3f prune_work=%.3f adj_nets=%d scan_nets=%d max_node_edge_product=%lld\n",
                     seconds_since(build_start),
                     finalize_wall_seconds,
                     final_pinloc_seconds,
                     final_attach_seconds,
                     final_reorder_seconds,
                     final_repair_seconds,
                     final_prune_seconds,
                     repair_adjacency_nets,
                     repair_scan_nets,
                     repair_node_edge_product_max);
        std::fflush(stderr);
    }

    graph.net2node_start.resize(static_cast<std::size_t>(num_nets) + 1, 0);
    graph.net2edge_start.resize(static_cast<std::size_t>(num_nets) + 1, 0);
    long long total_nodes = 0;
    long long total_edges = 0;
    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        total_nodes += net_node_count[net_idx];
        total_edges += net_edge_count[net_idx];
        if (total_nodes > std::numeric_limits<int>::max() ||
            total_edges > std::numeric_limits<int>::max()) {
            throw std::runtime_error("OpenROAD route-segment RC graph exceeds int indexing capacity.");
        }
        graph.net2node_start[net_idx + 1] = static_cast<int>(total_nodes);
        graph.net2edge_start[net_idx + 1] = static_cast<int>(total_edges);
    }
    graph.num_nodes = static_cast<int>(total_nodes);
    graph.num_edges = static_cast<int>(total_edges);

    const auto materialize_start = std::chrono::steady_clock::now();
    graph.edge_from.reserve(static_cast<std::size_t>(graph.num_edges));
    graph.edge_to.reserve(static_cast<std::size_t>(graph.num_edges));
    graph.edge_res.reserve(static_cast<std::size_t>(graph.num_edges));
    graph.node2pin.reserve(static_cast<std::size_t>(graph.num_nodes));
    graph.node_cap.reserve(static_cast<std::size_t>(graph.num_nodes) * NUM_ATTR);
    if (keep_route_node_names) {
        graph.node_names.reserve(static_cast<std::size_t>(graph.num_nodes));
    }

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        LocalSpefNetRc* local = local_nets[net_idx].get();
        if (local == nullptr) {
            continue;
        }
        const int node_base = graph.net2node_start[net_idx];
        for (std::size_t edge = 0; edge < local->edge_from.size(); ++edge) {
            graph.edge_from.emplace_back(node_base + local->edge_from[edge]);
            graph.edge_to.emplace_back(node_base + local->edge_to[edge]);
            graph.edge_res.emplace_back(local->edge_res[edge]);
        }
        for (int node = 0; node < static_cast<int>(local->node2pin.size()); ++node) {
            graph.node2pin.emplace_back(local->node2pin[node]);
            if (keep_route_node_names) {
                if (node < static_cast<int>(local->node_names.size())) {
                    graph.node_names.emplace_back(std::move(local->node_names[node]));
                } else {
                    graph.node_names.emplace_back("");
                }
            }
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                graph.node_cap.emplace_back(local->node_cap[node * NUM_ATTR + attr]);
            }
        }
        local_nets[net_idx].reset();
        route_node_maps[net_idx].reset();
    }
    local_nets.clear();
    route_node_maps.clear();
    final_append_seconds = seconds_since(materialize_start);

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=finalize_done elapsed=%.3f nodes=%d edges=%d pinloc=%.3f attach=%.3f reorder=%.3f repair=%.3f prune=%.3f append=%.3f adj_nets=%d scan_nets=%d max_node_edge_product=%lld skipped_missing_high_fanout_nets=%d skipped_missing_high_fanout_pins=%lld skipped_missing_unconnected_nets=%d skipped_missing_unconnected_pins=%lld\n",
                     seconds_since(build_start),
                     graph.num_nodes,
                     graph.num_edges,
                     final_pinloc_seconds,
                     final_attach_seconds,
                     final_reorder_seconds,
                     final_repair_seconds,
                     final_prune_seconds,
                     final_append_seconds,
                     repair_adjacency_nets,
                     repair_scan_nets,
                     repair_node_edge_product_max,
                     stats.skipped_missing_high_fanout_nets,
                     stats.skipped_missing_high_fanout_pins,
                     stats.skipped_missing_unconnected_nets,
                     stats.skipped_missing_unconnected_pins);
        std::fflush(stderr);
    }

    graph.skipped_loop_edges = stats.skipped_loop_edges;
    graph.repaired_edges = stats.repaired_edges;

    logger.info("OpenROAD route-segment RC graph: file=%s parsed_nets=%d missing_nets=%d unknown_nets=%d nodes=%d edges=%d",
                file.c_str(), stats.parsed_nets, stats.missing_nets,
                stats.unknown_nets, graph.num_nodes, graph.num_edges);
    logger.info("OpenROAD route-segment RC details: segment_rows=%d wires=%d vias=%d malformed=%d unknown_layers=%d non_manhattan=%d self_segments=%d pin_stub_edges=%d missing_driver_nodes=%d missing_net_pins=%d fallback_net_pins=%d skipped_missing_unconnected_nets=%d skipped_missing_unconnected_pins=%lld skipped_missing_high_fanout_nets=%d skipped_missing_high_fanout_pins=%lld repaired_edges=%d loop_edges=%d",
                stats.segment_rows, stats.wire_segments, stats.via_segments,
                stats.malformed_rows, stats.unknown_layers,
                stats.non_manhattan_segments, stats.skipped_self_segments,
                stats.pin_stub_edges, stats.missing_driver_nodes,
                stats.missing_net_pins, stats.fallback_net_pins,
                stats.skipped_missing_unconnected_nets,
                stats.skipped_missing_unconnected_pins,
                stats.skipped_missing_high_fanout_nets,
                stats.skipped_missing_high_fanout_pins,
                stats.repaired_edges, stats.skipped_loop_edges);

    if (stats.malformed_rows > 0 || stats.unknown_layers > 0 ||
        stats.non_manhattan_segments > 0) {
        std::ostringstream msg;
        msg << "OpenROAD route segment parser found unsupported rows: "
            << "malformed=" << stats.malformed_rows
            << " unknown_layers=" << stats.unknown_layers
            << " non_manhattan=" << stats.non_manhattan_segments
            << ". Check the raw route segment format before timing comparison.";
        throw std::runtime_error(msg.str());
    }

    if (cache_enabled && cache_meta.source_size > 0) {
        save_route_segment_cache(cache_path, cache_meta, num_nets, num_pins,
                                 missing_high_fanout_skip,
                                 cache_design_signature,
                                 graph);
    }

    return graph;
}

}  // namespace gt
