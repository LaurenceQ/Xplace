#include "gputimer/core/openroad/OpenroadRcInternal.h"

namespace gt {
namespace openroad_rc {

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

struct RouteSegmentCacheHeader {
    char magic[16];
    std::uint32_t version = 0;
    std::uint32_t reserved = 0;
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
    std::uint64_t design_signature = 0;
    std::int32_t num_nets = 0;
    std::int32_t num_pins = 0;
    std::int32_t missing_high_fanout_skip = 0;
    std::int32_t num_nodes = 0;
    std::int32_t num_edges = 0;
    std::int32_t skipped_loop_edges = 0;
    std::int32_t repaired_edges = 0;
};

constexpr char ROUTE_SEG_CACHE_MAGIC[16] = {
    'X', 'P', 'R', 'S', 'E', 'G', 'R', 'C', 'A', 'C', 'H', 'E', '0', '5', '\0', '\0'
};
constexpr std::uint32_t ROUTE_SEG_CACHE_VERSION = 5;

std::uint64_t fnv1a_mix(std::uint64_t hash, const void* data, std::size_t bytes) {
    const auto* ptr = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(ptr[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void fnv1a_mix_string(std::uint64_t& hash, const std::string& value) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    hash = fnv1a_mix(hash, &size, sizeof(size));
    hash = fnv1a_mix(hash, value.data(), value.size());
}

template <typename T>
void fnv1a_mix_value(std::uint64_t& hash, const T& value) {
    hash = fnv1a_mix(hash, &value, sizeof(value));
}

std::uint64_t route_segment_design_signature(const GTDatabase& gtdb,
                                             int expected_num_nets,
                                             int expected_num_pins) {
    std::uint64_t hash = 1469598103934665603ULL;
    fnv1a_mix_value(hash, expected_num_nets);
    fnv1a_mix_value(hash, expected_num_pins);
    fnv1a_mix_value(hash, gtdb.res_unit);
    fnv1a_mix_value(hash, gtdb.cap_unit);
    for (const std::string& name : gtdb.net_names) {
        fnv1a_mix_string(hash, name);
    }
    for (const std::string& name : gtdb.pin_names) {
        fnv1a_mix_string(hash, name);
    }
    return hash;
}

RouteSegmentCacheMeta route_segment_cache_meta(const std::string& source_file) {
    RouteSegmentCacheMeta meta;
    try {
        meta.source_size = static_cast<std::uint64_t>(std::filesystem::file_size(source_file));
        meta.source_mtime = static_cast<std::int64_t>(
            std::filesystem::last_write_time(source_file).time_since_epoch().count());
    } catch (const std::exception&) {
        meta = RouteSegmentCacheMeta{};
    }
    return meta;
}

std::string route_segment_cache_path(const std::string& source_file) {
    const char* cache_dir_env = std::getenv("GPUTIMER_ROUTE_SEG_CACHE_DIR");
    std::filesystem::path cache_dir =
        (cache_dir_env != nullptr && cache_dir_env[0] != '\0')
            ? std::filesystem::path(cache_dir_env)
            : std::filesystem::path("result") / "route_segment_cache";
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);

    const std::filesystem::path source_path(source_file);
    const std::string base = source_path.filename().string();
    std::ostringstream hash_stream;
    hash_stream << std::hex << std::hash<std::string>{}(std::filesystem::absolute(source_path).string());
    return (cache_dir / (base + "." + hash_stream.str() + ".rcgraph.v5.bin")).string();
}

template <typename T>
bool write_cache_vector(std::ofstream& output, const std::vector<T>& values) {
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(sizeof(T) * values.size()));
    }
    return static_cast<bool>(output);
}

template <typename T>
bool read_cache_vector(std::ifstream& input,
                       std::vector<T>& values,
                       std::uint64_t expected_size) {
    std::uint64_t size = 0;
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!input || size != expected_size) {
        return false;
    }
    values.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(sizeof(T) * values.size()));
    }
    return static_cast<bool>(input);
}

bool load_route_segment_cache(const std::string& cache_path,
                              const RouteSegmentCacheMeta& meta,
                              int expected_num_nets,
                              int expected_num_pins,
                              int missing_high_fanout_skip,
                              std::uint64_t expected_design_signature,
                              HostRcGraph& graph) {
    const bool profile = env_enabled("GPUTIMER_ROUTE_SEG_CACHE_PROFILE") ||
                         env_enabled("DMP_RC_PROFILE");
    const auto start = std::chrono::steady_clock::now();
    auto last = start;
    auto log_profile = [&](const char* phase) {
        if (!profile) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last).count();
        const double total = std::chrono::duration<double>(now - start).count();
        std::fprintf(stderr,
                     "[ROUTE_SEG_CACHE_PROFILE] phase=%s elapsed=%.3f total=%.3f path=%s\n",
                     phase,
                     elapsed,
                     total,
                     cache_path.c_str());
        std::fflush(stderr);
        last = now;
    };

    std::ifstream input(cache_path, std::ios::binary);
    if (!input) {
        return false;
    }
    log_profile("open_cache");

    RouteSegmentCacheHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input ||
        std::memcmp(header.magic, ROUTE_SEG_CACHE_MAGIC, sizeof(header.magic)) != 0 ||
        header.version != ROUTE_SEG_CACHE_VERSION ||
        header.source_size != meta.source_size ||
        header.source_mtime != meta.source_mtime ||
        header.design_signature != expected_design_signature ||
        header.num_nets != expected_num_nets ||
        header.num_pins != expected_num_pins ||
        header.missing_high_fanout_skip != missing_high_fanout_skip ||
        header.num_nodes < 0 ||
        header.num_edges < 0) {
        return false;
    }
    log_profile("read_header");

    HostRcGraph loaded;
    loaded.num_nodes = header.num_nodes;
    loaded.num_edges = header.num_edges;
    loaded.skipped_loop_edges = header.skipped_loop_edges;
    loaded.repaired_edges = header.repaired_edges;

    const std::uint64_t num_nets = static_cast<std::uint64_t>(expected_num_nets);
    const std::uint64_t num_nodes = static_cast<std::uint64_t>(loaded.num_nodes);
    const std::uint64_t num_edges = static_cast<std::uint64_t>(loaded.num_edges);
    if (!read_cache_vector(input, loaded.edge_from, num_edges)) {
        return false;
    }
    log_profile("read_edge_from");
    if (!read_cache_vector(input, loaded.edge_to, num_edges)) {
        return false;
    }
    log_profile("read_edge_to");
    if (!read_cache_vector(input, loaded.edge_res, num_edges)) {
        return false;
    }
    log_profile("read_edge_res");
    if (!read_cache_vector(input, loaded.node_cap, num_nodes * NUM_ATTR)) {
        return false;
    }
    log_profile("read_node_cap");
    if (!read_cache_vector(input, loaded.net2node_start, num_nets + 1)) {
        return false;
    }
    log_profile("read_net2node_start");
    if (!read_cache_vector(input, loaded.net2edge_start, num_nets + 1)) {
        return false;
    }
    log_profile("read_net2edge_start");
    if (!read_cache_vector(input, loaded.node2pin, num_nodes)) {
        return false;
    }
    log_profile("read_node2pin");
    if (!read_cache_vector(input, loaded.includes_pin_caps, num_nets)) {
        return false;
    }
    log_profile("read_includes_pin_caps");

    graph = std::move(loaded);
    log_profile("load_done");
    return true;
}

void save_route_segment_cache(const std::string& cache_path,
                              const RouteSegmentCacheMeta& meta,
                              int expected_num_nets,
                              int expected_num_pins,
                              int missing_high_fanout_skip,
                              std::uint64_t design_signature,
                              const HostRcGraph& graph) {
    const std::string tmp_path = cache_path + ".tmp";
    std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return;
    }

    RouteSegmentCacheHeader header;
    std::memcpy(header.magic, ROUTE_SEG_CACHE_MAGIC, sizeof(header.magic));
    header.version = ROUTE_SEG_CACHE_VERSION;
    header.source_size = meta.source_size;
    header.source_mtime = meta.source_mtime;
    header.design_signature = design_signature;
    header.num_nets = expected_num_nets;
    header.num_pins = expected_num_pins;
    header.missing_high_fanout_skip = missing_high_fanout_skip;
    header.num_nodes = graph.num_nodes;
    header.num_edges = graph.num_edges;
    header.skipped_loop_edges = graph.skipped_loop_edges;
    header.repaired_edges = graph.repaired_edges;
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!output ||
        !write_cache_vector(output, graph.edge_from) ||
        !write_cache_vector(output, graph.edge_to) ||
        !write_cache_vector(output, graph.edge_res) ||
        !write_cache_vector(output, graph.node_cap) ||
        !write_cache_vector(output, graph.net2node_start) ||
        !write_cache_vector(output, graph.net2edge_start) ||
        !write_cache_vector(output, graph.node2pin) ||
        !write_cache_vector(output, graph.includes_pin_caps)) {
        output.close();
        std::error_code remove_ec;
        std::filesystem::remove(tmp_path, remove_ec);
        return;
    }
    output.close();
    std::error_code ec;
    std::filesystem::rename(tmp_path, cache_path, ec);
    if (ec) {
        std::filesystem::remove(cache_path, ec);
        ec.clear();
        std::filesystem::rename(tmp_path, cache_path, ec);
    }
}

}  // namespace openroad_rc
}  // namespace gt
