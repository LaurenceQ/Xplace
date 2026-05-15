#include "GPUTimer.h"
#include "common/utils/utils.h"
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
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {

namespace {

static bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

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
    std::ifstream input(cache_path, std::ios::binary);
    if (!input) {
        return false;
    }

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

    HostRcGraph loaded;
    loaded.num_nodes = header.num_nodes;
    loaded.num_edges = header.num_edges;
    loaded.skipped_loop_edges = header.skipped_loop_edges;
    loaded.repaired_edges = header.repaired_edges;

    const std::uint64_t num_nets = static_cast<std::uint64_t>(expected_num_nets);
    const std::uint64_t num_nodes = static_cast<std::uint64_t>(loaded.num_nodes);
    const std::uint64_t num_edges = static_cast<std::uint64_t>(loaded.num_edges);
    if (!read_cache_vector(input, loaded.edge_from, num_edges) ||
        !read_cache_vector(input, loaded.edge_to, num_edges) ||
        !read_cache_vector(input, loaded.edge_res, num_edges) ||
        !read_cache_vector(input, loaded.node_cap, num_nodes * NUM_ATTR) ||
        !read_cache_vector(input, loaded.net2node_start, num_nets + 1) ||
        !read_cache_vector(input, loaded.net2edge_start, num_nets + 1) ||
        !read_cache_vector(input, loaded.node2pin, num_nodes) ||
        !read_cache_vector(input, loaded.includes_pin_caps, num_nets)) {
        return false;
    }

    graph = std::move(loaded);
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

struct LocalSpefNetRc {
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

struct OpenroadGrRcBuildStats {
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

struct OpenroadRouteSegmentsBuildStats {
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
};

struct OpenroadInferredGrid {
    int tile_size = 0;
    int origin_x = 0;
    int origin_y = 0;
    bool valid = false;
};

struct NangateLayerRc {
    float res_ohm_per_um = 0.0f;
    float cap_f_per_um = 0.0f;
};

static std::string normalized_spef_name(std::string name)
{
    validate_token(name);
    return name;
}

static bool spef_digits_only(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

static void add_name_alias(std::unordered_map<std::string, int>& map,
                           const std::string& name,
                           int value)
{
    if (name.empty()) {
        return;
    }
    map.emplace(name, value);
    map.emplace(normalized_spef_name(name), value);
}

static std::string replace_char(std::string name, char from, char to)
{
    std::replace(name.begin(), name.end(), from, to);
    return name;
}

static std::string replace_last_char(std::string name, char from, char to)
{
    const std::size_t pos = name.rfind(from);
    if (pos != std::string::npos) {
        name[pos] = to;
    }
    return name;
}

static void add_gr_name_alias(std::unordered_map<std::string, int>& map,
                              const std::string& name,
                              int value)
{
    add_name_alias(map, name, value);
    add_name_alias(map, replace_char(name, '/', ':'), value);
    add_name_alias(map, replace_char(name, ':', '/'), value);
    add_name_alias(map, replace_last_char(name, '/', ':'), value);
    add_name_alias(map, replace_last_char(name, ':', '/'), value);
}

static void add_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] += cap;
    }
}

static void set_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] = cap;
    }
}

static void set_attr_cap(std::vector<float>& node_cap, int node, int attr, float cap)
{
    if (attr >= 0 && attr < NUM_ATTR) {
        node_cap[node * NUM_ATTR + attr] = cap;
    }
}

static float pin_cap_attr_host(const GTDatabase& gtdb, int pin, int attr)
{
    if (pin < 0 || attr < 0 || attr >= NUM_ATTR) {
        return 0.0f;
    }
    const int stride = NUM_ATTR + 2;
    const int base = pin * stride;
    if (base + stride > static_cast<int>(gtdb.pin_capacitance.size())) {
        return 0.0f;
    }
    float cap = gtdb.pin_capacitance[base + attr];
    if (std::isfinite(cap)) {
        return cap;
    }
    cap = gtdb.pin_capacitance[base + NUM_ATTR + (attr >> 1)];
    return std::isfinite(cap) ? cap : 0.0f;
}

static std::string openroad_gr_edge_key(int from, int to, const std::string& res_id)
{
    return std::to_string(from) + '\t' + std::to_string(to) + '\t' + res_id;
}

static std::vector<std::string> split_tsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        std::size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            fields.emplace_back(line.substr(begin));
            break;
        }
        fields.emplace_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return fields;
}

static bool parse_int_field(const std::string& value, int& out)
{
    try {
        std::size_t consumed = 0;
        out = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

static const char* skip_route_ws(const char* ptr, const char* end)
{
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr))) {
        ++ptr;
    }
    return ptr;
}

static bool parse_int_token(const char*& ptr, const char* end, int& out)
{
    ptr = skip_route_ws(ptr, end);
    if (ptr >= end) {
        return false;
    }
    bool negative = false;
    if (*ptr == '+' || *ptr == '-') {
        negative = *ptr == '-';
        ++ptr;
    }
    if (ptr >= end || !std::isdigit(static_cast<unsigned char>(*ptr))) {
        return false;
    }
    long long value = 0;
    while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) {
        value = value * 10 + static_cast<int>(*ptr - '0');
        if ((!negative && value > std::numeric_limits<int>::max()) ||
            (negative && -value < std::numeric_limits<int>::min())) {
            return false;
        }
        ++ptr;
    }
    out = static_cast<int>(negative ? -value : value);
    return true;
}

static bool parse_token_range(const char*& ptr,
                              const char* end,
                              const char*& begin,
                              const char*& finish)
{
    ptr = skip_route_ws(ptr, end);
    begin = ptr;
    while (ptr < end && !std::isspace(static_cast<unsigned char>(*ptr))) {
        ++ptr;
    }
    finish = ptr;
    return begin < finish;
}

static bool route_rest_is_ws(const char* ptr, const char* end)
{
    return skip_route_ws(ptr, end) == end;
}

static bool parse_route_segment_row(const std::string& line,
                                    int& x1,
                                    int& y1,
                                    const char*& layer1_begin,
                                    const char*& layer1_end,
                                    int& x2,
                                    int& y2,
                                    const char*& layer2_begin,
                                    const char*& layer2_end)
{
    const char* ptr = line.data();
    const char* end = ptr + line.size();
    if (!parse_int_token(ptr, end, x1) ||
        !parse_int_token(ptr, end, y1) ||
        !parse_token_range(ptr, end, layer1_begin, layer1_end) ||
        !parse_int_token(ptr, end, x2) ||
        !parse_int_token(ptr, end, y2) ||
        !parse_token_range(ptr, end, layer2_begin, layer2_end)) {
        return false;
    }
    return route_rest_is_ws(ptr, end);
}

static bool parse_float_field(const std::string& value, float& out)
{
    try {
        std::size_t consumed = 0;
        out = std::stof(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

static std::vector<std::string> split_whitespace(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (stream >> field) {
        fields.emplace_back(field);
    }
    return fields;
}

static std::string lowercase_string(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static int trailing_integer(const std::string& value)
{
    std::size_t pos = value.size();
    while (pos > 0 && std::isdigit(static_cast<unsigned char>(value[pos - 1]))) {
        --pos;
    }
    if (pos == value.size()) {
        return -1;
    }
    int parsed = -1;
    return parse_int_field(value.substr(pos), parsed) ? parsed : -1;
}

static int trailing_integer_token(const char* begin, const char* end)
{
    const char* digits = end;
    while (digits > begin && std::isdigit(static_cast<unsigned char>(*(digits - 1)))) {
        --digits;
    }
    if (digits == end) {
        return -1;
    }
    const char* ptr = digits;
    int value = -1;
    return parse_int_token(ptr, end, value) && ptr == end ? value : -1;
}

static bool route_point_matches(const OpenroadRoutePt& pt, const OpenroadRoutePtKey& key)
{
    return pt.valid && pt.x == key.x && pt.y == key.y && pt.layer == key.layer;
}

static int openroad_layer_track_spacing(const db::Layer& layer)
{
    int spacing = 0;
    for (const db::Track& track : layer.tracks) {
        if (track.step > 0) {
            spacing = static_cast<int>(track.step);
            break;
        }
    }
    if (spacing <= 0) {
        spacing = layer.pitch;
    }
    if (spacing > 0 && (spacing % 2) != 0) {
        --spacing;
    }
    return spacing;
}

static int openroad_gcell_tile_size(const db::Database& rawdb)
{
    constexpr int upper_layer_for_gcell_size = 4;
    constexpr int pitches_in_tile = 15;

    int max_routing_level = 0;
    for (const db::Layer& layer : rawdb.layers) {
        if (layer.rIndex >= 0) {
            max_routing_level = std::max(max_routing_level, layer.rIndex + 1);
        }
    }
    if (max_routing_level <= 0) {
        return 0;
    }

    auto spacing_for_level = [&](int routing_level) {
        for (const db::Layer& layer : rawdb.layers) {
            if (layer.rIndex + 1 == routing_level) {
                return openroad_layer_track_spacing(layer);
            }
        }
        return 0;
    };

    if (max_routing_level < upper_layer_for_gcell_size) {
        return spacing_for_level(max_routing_level) * pitches_in_tile;
    }

    std::vector<int> track_spacings;
    for (int level = 2; level <= 4; ++level) {
        const int spacing = spacing_for_level(level);
        if (spacing > 0) {
            track_spacings.emplace_back(spacing);
        }
    }
    if (track_spacings.empty()) {
        return 0;
    }
    std::sort(track_spacings.begin(), track_spacings.end());
    return track_spacings[track_spacings.size() / 2] * pitches_in_tile;
}

static std::pair<int, int> openroad_position_on_grid(const db::Database& rawdb,
                                                     int tile_size,
                                                     int x,
                                                     int y)
{
    const int x_grids = std::max(1, (rawdb.dieHX - rawdb.dieLX) / tile_size);
    const int y_grids = std::max(1, (rawdb.dieHY - rawdb.dieLY) / tile_size);
    int gcell_id_x = (x - rawdb.dieLX) / tile_size;
    int gcell_id_y = (y - rawdb.dieLY) / tile_size;

    if (gcell_id_x >= x_grids) {
        --gcell_id_x;
    }
    if (gcell_id_y >= y_grids) {
        --gcell_id_y;
    }
    gcell_id_x = std::max(0, gcell_id_x);
    gcell_id_y = std::max(0, gcell_id_y);

    return {gcell_id_x * tile_size + tile_size / 2 + rawdb.dieLX,
            gcell_id_y * tile_size + tile_size / 2 + rawdb.dieLY};
}

static int positive_mod(int value, int modulus)
{
    if (modulus <= 0) {
        return 0;
    }
    int result = value % modulus;
    if (result < 0) {
        result += modulus;
    }
    return result;
}

static int infer_openroad_grid_step(std::vector<int> coords)
{
    if (coords.size() < 2) {
        return 0;
    }
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    int step = 0;
    for (std::size_t i = 1; i < coords.size(); ++i) {
        const int diff = coords[i] - coords[i - 1];
        if (diff > 0) {
            step = std::gcd(step, diff);
        }
    }
    return step;
}

static int infer_openroad_grid_origin(const std::vector<int>& coords,
                                      int tile_size,
                                      int fallback_origin)
{
    if (coords.empty() || tile_size <= 0) {
        return fallback_origin;
    }
    int origin = positive_mod(coords.front(), tile_size) - tile_size / 2;
    const int delta = fallback_origin - origin;
    int shift = delta / tile_size;
    int best_origin = origin + shift * tile_size;
    if (std::abs((best_origin + tile_size) - fallback_origin) <
        std::abs(best_origin - fallback_origin)) {
        best_origin += tile_size;
    }
    if (std::abs((best_origin - tile_size) - fallback_origin) <
        std::abs(best_origin - fallback_origin)) {
        best_origin -= tile_size;
    }
    return best_origin;
}

static int infer_openroad_grid_origin_from_first(int first_coord,
                                                bool have_coord,
                                                int tile_size,
                                                int fallback_origin)
{
    if (!have_coord || tile_size <= 0) {
        return fallback_origin;
    }
    int origin = positive_mod(first_coord, tile_size) - tile_size / 2;
    const int delta = fallback_origin - origin;
    int shift = delta / tile_size;
    int best_origin = origin + shift * tile_size;
    if (std::abs((best_origin + tile_size) - fallback_origin) <
        std::abs(best_origin - fallback_origin)) {
        best_origin += tile_size;
    }
    if (std::abs((best_origin - tile_size) - fallback_origin) <
        std::abs(best_origin - fallback_origin)) {
        best_origin -= tile_size;
    }
    return best_origin;
}

static OpenroadInferredGrid infer_openroad_route_grid(
    const std::vector<std::unique_ptr<LocalSpefNetRc>>& local_nets,
    const db::Database& rawdb,
    int fallback_tile_size)
{
    bool have_x = false;
    bool have_y = false;
    int first_x = 0;
    int first_y = 0;
    int x_step = 0;
    int y_step = 0;
    for (const auto& local_ptr : local_nets) {
        if (!local_ptr) {
            continue;
        }
        for (const OpenroadRoutePt& pt : local_ptr->route_points) {
            if (pt.valid) {
                if (!have_x) {
                    have_x = true;
                    first_x = pt.x;
                } else {
                    x_step = std::gcd(x_step, std::abs(pt.x - first_x));
                }
                if (!have_y) {
                    have_y = true;
                    first_y = pt.y;
                } else {
                    y_step = std::gcd(y_step, std::abs(pt.y - first_y));
                }
            }
        }
    }

    int tile_size = std::gcd(x_step, y_step);
    if (tile_size <= 0) {
        tile_size = std::max(x_step, y_step);
    }
    if (tile_size <= 0) {
        tile_size = fallback_tile_size;
    }

    OpenroadInferredGrid grid;
    grid.tile_size = tile_size;
    grid.origin_x = infer_openroad_grid_origin_from_first(first_x, have_x, tile_size, rawdb.dieLX);
    grid.origin_y = infer_openroad_grid_origin_from_first(first_y, have_y, tile_size, rawdb.dieLY);
    grid.valid = tile_size > 0;
    return grid;
}

static std::pair<int, int> openroad_position_on_inferred_grid(
    const db::Database& rawdb,
    const OpenroadInferredGrid& grid,
    int x,
    int y)
{
    if (!grid.valid || grid.tile_size <= 0) {
        return {x, y};
    }

    const int tile_size = grid.tile_size;
    const int x_grids = std::max(1, (rawdb.dieHX - rawdb.dieLX) / tile_size);
    const int y_grids = std::max(1, (rawdb.dieHY - rawdb.dieLY) / tile_size);
    int gcell_id_x = (x - grid.origin_x) / tile_size;
    int gcell_id_y = (y - grid.origin_y) / tile_size;

    if (gcell_id_x >= x_grids) {
        --gcell_id_x;
    }
    if (gcell_id_y >= y_grids) {
        --gcell_id_y;
    }
    gcell_id_x = std::max(0, gcell_id_x);
    gcell_id_y = std::max(0, gcell_id_y);

    return {gcell_id_x * tile_size + tile_size / 2 + grid.origin_x,
            gcell_id_y * tile_size + tile_size / 2 + grid.origin_y};
}

static NangateLayerRc nangate45_layer_rc(int routing_level)
{
    static const std::array<NangateLayerRc, 11> rc = {{
        {0.0f, 0.0f},
        {5.4286e-03f * 1.0e3f, 7.41819e-02f * 1.0e-15f},
        {3.5714e-03f * 1.0e3f, 6.74606e-02f * 1.0e-15f},
        {3.5714e-03f * 1.0e3f, 8.88758e-02f * 1.0e-15f},
        {1.5000e-03f * 1.0e3f, 1.07121e-01f * 1.0e-15f},
        {1.5000e-03f * 1.0e3f, 1.08964e-01f * 1.0e-15f},
        {1.5000e-03f * 1.0e3f, 1.02044e-01f * 1.0e-15f},
        {1.8750e-04f * 1.0e3f, 1.10436e-01f * 1.0e-15f},
        {1.8750e-04f * 1.0e3f, 9.69714e-02f * 1.0e-15f},
        {3.7500e-05f * 1.0e3f, 3.6864e-02f * 1.0e-15f},
        {3.7500e-05f * 1.0e3f, 2.8042e-02f * 1.0e-15f},
    }};
    if (routing_level <= 0 || routing_level >= static_cast<int>(rc.size())) {
        return {};
    }
    return rc[routing_level];
}

static float nangate45_via_res_ohm(int lower_routing_level)
{
    static const std::array<float, 10> via_res = {{
        0.0f,
        5.0f,
        5.0f,
        5.0f,
        3.0f,
        3.0f,
        3.0f,
        1.0f,
        1.0f,
        0.5f,
    }};
    if (lower_routing_level <= 0 ||
        lower_routing_level >= static_cast<int>(via_res.size())) {
        return 0.0f;
    }
    return via_res[lower_routing_level];
}

static std::tuple<int, int, int, int> orient_box_for_iopin(int orient,
                                                           int lx,
                                                           int ly,
                                                           int hx,
                                                           int hy)
{
    switch (orient) {
        case 1:  // W
            return {-hy, lx, -ly, hx};
        case 2:  // S
            return {-hx, -hy, -lx, -ly};
        case 3:  // E
            return {ly, -hx, hy, -lx};
        case 4:  // FN
            return {-hx, ly, -lx, hy};
        case 5:  // FW
            return {ly, lx, hy, hx};
        case 6:  // FS
            return {lx, -hy, hx, -ly};
        case 7:  // FE
            return {-hy, -hx, -ly, -lx};
        default:
            return {lx, ly, hx, hy};
    }
}

static std::tuple<int, int, int, int> orient_box_for_cell(const db::CellType* ctype,
                                                          int orient,
                                                          int lx,
                                                          int ly,
                                                          int hx,
                                                          int hy)
{
    if (ctype == nullptr) {
        return {lx, ly, hx, hy};
    }
    switch (orient) {
        case 2:  // S
            return {ctype->width - hx,
                    ctype->height - hy,
                    ctype->width - lx,
                    ctype->height - ly};
        case 4:  // FN
            return {ctype->width - hx, ly, ctype->width - lx, hy};
        case 6:  // FS
            return {lx, ctype->height - hy, hx, ctype->height - ly};
        default:
            return {lx, ly, hx, hy};
    }
}

static void ensure_local_node(LocalSpefNetRc& local, int node_id)
{
    while (node_id >= static_cast<int>(local.node2pin.size())) {
        local.node2pin.emplace_back(-1);
        local.node_names.emplace_back("");
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
    }
}

static std::string spef_upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

static bool spef_includes_pin_caps_from_design_flow(const std::string& design_flow)
{
    std::string flow = spef_upper(design_flow);
    size_t pin_cap_pos = flow.find("PIN_CAP");
    if (pin_cap_pos == std::string::npos) {
        return false;
    }
    std::string pin_cap_clause = flow.substr(pin_cap_pos);
    return pin_cap_clause.find("NONE") == std::string::npos;
}

static int count_tree_edges_from_root(const LocalSpefNetRc& local)
{
    if (local.node2pin.empty()) {
        return 0;
    }
    std::vector<uint8_t> seen(local.node2pin.size(), 0);
    std::vector<int> stack;
    int tree_edges = 0;
    seen[0] = 1;
    stack.emplace_back(0);
    for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
        int node = stack[cursor];
        for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            int from = local.edge_from[edge];
            int to = local.edge_to[edge];
            int next = -1;
            if (from == node) next = to;
            if (to == node) next = from;
            if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next);
                tree_edges++;
            }
        }
    }
    return tree_edges;
}

static int prune_to_rooted_tree(LocalSpefNetRc& local)
{
    if (local.node2pin.empty() || local.edge_from.empty()) {
        return 0;
    }

    const int node_count = static_cast<int>(local.node2pin.size());
    std::vector<uint8_t> seen(node_count, 0);
    std::vector<uint8_t> keep(local.edge_from.size(), 0);
    std::vector<int> stack;
    seen[0] = 1;
    stack.emplace_back(0);

    for (std::size_t cursor = 0; cursor < stack.size(); ++cursor) {
        const int node = stack[cursor];
        for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            const int from = local.edge_from[edge];
            const int to = local.edge_to[edge];
            int next = -1;
            if (from == node) next = to;
            if (to == node) next = from;
            if (next >= 0 && next < node_count && !seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next);
                keep[edge] = 1;
            }
        }
    }

    int kept_edges = 0;
    for (uint8_t keep_edge : keep) {
        if (keep_edge) {
            kept_edges++;
        }
    }
    const int dropped_edges = static_cast<int>(local.edge_from.size()) - kept_edges;
    if (dropped_edges == 0) {
        return 0;
    }

    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    edge_from.reserve(kept_edges);
    edge_to.reserve(kept_edges);
    edge_res.reserve(kept_edges);
    for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
        if (!keep[edge]) {
            continue;
        }
        edge_from.emplace_back(local.edge_from[edge]);
        edge_to.emplace_back(local.edge_to[edge]);
        edge_res.emplace_back(local.edge_res[edge]);
    }
    local.edge_from.swap(edge_from);
    local.edge_to.swap(edge_to);
    local.edge_res.swap(edge_res);
    return dropped_edges;
}

}  // namespace

HostRcGraph GPUTimer::build_openroad_gr_rc(const std::string& file) {
    std::ifstream input(file);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open OpenROAD GR RC dump: " + file);
    }

    torch::Tensor flat_net2pin_start_map_at =
        timing_raw_db.flat_net2pin_start_map.cpu().contiguous();
    torch::Tensor flat_net2pin_map_at =
        timing_raw_db.flat_net2pin_map.cpu().contiguous();
    torch::Tensor pin2net_map_at =
        timing_raw_db.pin2net_map.cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();
    const int* pin2net_map = pin2net_map_at.data_ptr<int>();

    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    std::unordered_map<std::string, int> global_pin_name_to_id;
    for (int i = 0; i < static_cast<int>(gtdb.pin_names.size()); ++i) {
        add_gr_name_alias(global_pin_name_to_id, gtdb.pin_names[i], i);
    }

    std::vector<LocalSpefNetRc> parsed_nets(num_nets);
    std::vector<std::unordered_map<std::string, int>> edge_key_to_id(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);
    OpenroadGrRcBuildStats stats;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> gr_corners;
    std::unordered_map<std::string, int> gr_corner_order;

    auto resolve_net = [&](const std::string& name) {
        auto iter = net_name_to_index.find(name);
        if (iter != net_name_to_index.end()) {
            return iter->second;
        }
        std::string normalized = normalized_spef_name(name);
        iter = net_name_to_index.find(normalized);
        return iter == net_name_to_index.end() ? -1 : iter->second;
    };

    auto resolve_pin = [&](const std::string& name) {
        auto iter = global_pin_name_to_id.find(name);
        if (iter != global_pin_name_to_id.end()) {
            return iter->second;
        }
        std::string normalized = normalized_spef_name(name);
        iter = global_pin_name_to_id.find(normalized);
        return iter == global_pin_name_to_id.end() ? -1 : iter->second;
    };

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> fields = split_tsv(line);
        if (fields.size() < 3) {
            continue;
        }

        if (fields[0] == "WARN") {
            continue;
        }
        if (gr_corner_order.find(fields[1]) == gr_corner_order.end()) {
            const int order = static_cast<int>(gr_corners.size());
            gr_corner_order.emplace(fields[1], order);
            gr_corners.emplace_back(fields[1]);
        }
        rows.emplace_back(std::move(fields));
    }

    if (gr_corners.size() != 1) {
        throw std::runtime_error(
            "OpenROAD GR RC dump must contain exactly one OpenSTA scene for this DMP flow: " + file);
    }
    logger.info("OpenROAD GR RC scene: %s (wire RC is scalar and copied to all DMP attrs)",
                gr_corners[0].c_str());

    for (const std::vector<std::string>& fields : rows) {
        const std::string& tag = fields[0];

        const int net_idx = resolve_net(fields[2]);
        if (net_idx < 0 || net_idx >= num_nets) {
            stats.unknown_nets++;
            continue;
        }
        if (!parsed_net[net_idx]) {
            parsed_net[net_idx] = 1;
            stats.parsed_nets++;
        }
        LocalSpefNetRc& local = parsed_nets[net_idx];

        if (tag == "NET") {
            continue;
        }

        if (tag == "NODE") {
            if (fields.size() < 10) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            float cap_f = 0.0f;
            if (!parse_int_field(fields[3], node_id) ||
                !parse_float_field(fields[9], cap_f) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            if (fields[4] == "route") {
                local.node_names[node_id] =
                    fields[6] + "," + fields[7] + ",M" + fields[8];
            } else {
                local.node_names[node_id] = fields[5];
            }
            set_attr_cap(local.node_cap, node_id, cap_f / gtdb.cap_unit);
            if (fields[4] == "pin") {
                const int pin_id = resolve_pin(fields[5]);
                if (pin_id >= 0 && pin2net_map[pin_id] == net_idx) {
                    local.node2pin[node_id] = pin_id;
                } else if (pin_id >= 0) {
                    stats.pin_net_mismatches++;
                } else if (!fields[5].empty()) {
                    stats.unresolved_pin_nodes++;
                }
            }
            stats.node_rows++;
            continue;
        }

        if (tag == "PIN") {
            if (fields.size() < 5) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            if (!parse_int_field(fields[3], node_id) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            local.node_names[node_id] = fields[4];
            const int pin_id = resolve_pin(fields[4]);
            if (pin_id >= 0 && pin2net_map[pin_id] == net_idx) {
                local.node2pin[node_id] = pin_id;
            } else if (pin_id >= 0) {
                stats.pin_net_mismatches++;
            } else if (!fields[4].empty()) {
                stats.unresolved_pin_nodes++;
            }
            stats.pin_rows++;
            continue;
        }

        if (tag == "CAP") {
            if (fields.size() < 5) {
                stats.malformed_rows++;
                continue;
            }
            int node_id = -1;
            float cap_f = 0.0f;
            if (!parse_int_field(fields[3], node_id) ||
                !parse_float_field(fields[4], cap_f) || node_id < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, node_id);
            set_attr_cap(local.node_cap, node_id, cap_f / gtdb.cap_unit);
            stats.cap_rows++;
            continue;
        }

        if (tag == "RES") {
            if (fields.size() < 8) {
                stats.malformed_rows++;
                continue;
            }
            int from = -1;
            int to = -1;
            float res_ohm = 0.0f;
            if (!parse_int_field(fields[5], from) ||
                !parse_int_field(fields[6], to) ||
                !parse_float_field(fields[7], res_ohm) ||
                from < 0 || to < 0) {
                stats.malformed_rows++;
                continue;
            }
            ensure_local_node(local, from);
            ensure_local_node(local, to);
            if (from == to) {
                stats.skipped_self_resistors++;
                continue;
            }
            auto& edge_map = edge_key_to_id[net_idx];
            const std::string edge_key = openroad_gr_edge_key(from, to, fields[3]);
            int edge_id = -1;
            auto edge_iter = edge_map.find(edge_key);
            if (edge_iter == edge_map.end()) {
                edge_id = static_cast<int>(local.edge_from.size());
                edge_map.emplace(edge_key, edge_id);
                local.edge_from.emplace_back(from);
                local.edge_to.emplace_back(to);
                local.edge_res.emplace_back(0.0f);
            } else {
                edge_id = edge_iter->second;
            }
            local.edge_res[edge_id] = res_ohm / gtdb.res_unit;
            stats.resistors++;
            continue;
        }
    }

    HostRcGraph graph;
    graph.includes_pin_caps.assign(num_nets, 0);
    graph.net2node_start.emplace_back(0);
    graph.net2edge_start.emplace_back(0);

    auto append_pin_node = [&](LocalSpefNetRc& local, int pin_id) {
        const int node_id = static_cast<int>(local.node2pin.size());
        local.node2pin.emplace_back(pin_id);
        local.node_names.emplace_back(pin_id >= 0 &&
                                      pin_id < static_cast<int>(gtdb.pin_names.size())
                                          ? gtdb.pin_names[pin_id]
                                          : "");
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
        return node_id;
    };

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        LocalSpefNetRc local;
        const int pin_begin = flat_net2pin_start_map[net_idx];
        const int pin_end = flat_net2pin_start_map[net_idx + 1];
        const int driver_pin = pin_begin < pin_end ? flat_net2pin_map[pin_begin] : -1;

        if (parsed_net[net_idx] && driver_pin >= 0) {
            const LocalSpefNetRc& source = parsed_nets[net_idx];
            std::vector<int> old_to_new(source.node2pin.size(), -1);
            int driver_old_node = -1;
            for (int node = 0; node < static_cast<int>(source.node2pin.size()); ++node) {
                if (source.node2pin[node] == driver_pin) {
                    driver_old_node = node;
                    break;
                }
            }

            auto append_old_node = [&](int old_node) {
                const int new_node = static_cast<int>(local.node2pin.size());
                old_to_new[old_node] = new_node;
                local.node2pin.emplace_back(source.node2pin[old_node]);
                local.node_names.emplace_back(source.node_names[old_node]);
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    local.node_cap.emplace_back(source.node_cap[old_node * NUM_ATTR + attr]);
                }
            };

            if (driver_old_node >= 0) {
                append_old_node(driver_old_node);
            } else {
                append_pin_node(local, driver_pin);
                stats.missing_driver_nodes++;
            }

            for (int old_node = 0; old_node < static_cast<int>(source.node2pin.size()); ++old_node) {
                if (old_node != driver_old_node) {
                    append_old_node(old_node);
                }
            }

            for (std::size_t edge = 0; edge < source.edge_from.size(); ++edge) {
                const int old_from = source.edge_from[edge];
                const int old_to = source.edge_to[edge];
                if (old_from < 0 || old_to < 0 ||
                    old_from >= static_cast<int>(old_to_new.size()) ||
                    old_to >= static_cast<int>(old_to_new.size())) {
                    continue;
                }
                const int from = old_to_new[old_from];
                const int to = old_to_new[old_to];
                if (from < 0 || to < 0 || from == to) {
                    stats.skipped_self_resistors++;
                    continue;
                }
                local.edge_from.emplace_back(from);
                local.edge_to.emplace_back(to);
                local.edge_res.emplace_back(source.edge_res[edge]);
            }
        } else if (!parsed_net[net_idx]) {
            stats.missing_nets++;
        }

        if (local.node2pin.empty() && driver_pin >= 0) {
            append_pin_node(local, driver_pin);
        }

        std::unordered_set<int> present_pins;
        for (int pin : local.node2pin) {
            if (pin >= 0) {
                present_pins.insert(pin);
            }
        }
        for (int pin_pos = pin_begin; pin_pos < pin_end; ++pin_pos) {
            const int pin_id = flat_net2pin_map[pin_pos];
            if (present_pins.insert(pin_id).second) {
                const int node = append_pin_node(local, pin_id);
                if (node > 0) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    stats.repaired_edges++;
                }
                if (parsed_net[net_idx]) {
                    stats.missing_net_pins++;
                } else {
                    stats.fallback_net_pins++;
                }
            }
        }

        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);
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
            for (int node = 1; node < static_cast<int>(seen.size()); ++node) {
                if (!seen[node]) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    stats.repaired_edges++;
                }
            }
        }

        const int skipped_loop_edges = prune_to_rooted_tree(local);
        if (skipped_loop_edges > 0) {
            stats.skipped_loop_edges += skipped_loop_edges;
        }

        for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            graph.edge_from.emplace_back(graph.num_nodes + local.edge_from[edge]);
            graph.edge_to.emplace_back(graph.num_nodes + local.edge_to[edge]);
            graph.edge_res.emplace_back(local.edge_res[edge]);
            graph.num_edges++;
        }
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            graph.node2pin.emplace_back(local.node2pin[node]);
            graph.node_names.emplace_back(local.node_names[node]);
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                graph.node_cap.emplace_back(local.node_cap[node * NUM_ATTR + attr]);
            }
        }
        graph.num_nodes += static_cast<int>(local.node2pin.size());
        graph.net2node_start.emplace_back(graph.num_nodes);
        graph.net2edge_start.emplace_back(graph.num_edges);
    }

    graph.skipped_loop_edges = stats.skipped_loop_edges;
    graph.repaired_edges = stats.repaired_edges;

    logger.info("OpenROAD GR RC graph: file=%s parsed_nets=%d missing_nets=%d unknown_nets=%d nodes=%d edges=%d",
                file.c_str(), stats.parsed_nets,
                stats.missing_nets, stats.unknown_nets, graph.num_nodes,
                graph.num_edges);
    logger.info("OpenROAD GR RC details: node_rows=%d pin_rows=%d cap_rows=%d resistors=%d self_res=%d unresolved_pin_nodes=%d pin_net_mismatches=%d missing_driver_nodes=%d missing_net_pins=%d fallback_net_pins=%d repaired_edges=%d loop_edges=%d malformed_rows=%d",
                stats.node_rows, stats.pin_rows, stats.cap_rows, stats.resistors,
                stats.skipped_self_resistors, stats.unresolved_pin_nodes,
                stats.pin_net_mismatches, stats.missing_driver_nodes, stats.missing_net_pins,
                stats.fallback_net_pins, stats.repaired_edges, stats.skipped_loop_edges,
                stats.malformed_rows);

    if (stats.unknown_nets > 0 || stats.malformed_rows > 0 ||
        stats.skipped_self_resistors > 0 ||
        stats.unresolved_pin_nodes > 0 || stats.pin_net_mismatches > 0 ||
        stats.missing_driver_nodes > 0 || stats.missing_net_pins > 0) {
        std::ostringstream msg;
        msg << "OpenROAD GR RC dump cannot be used for semantic timing alignment: "
            << "missing_nets=" << stats.missing_nets
            << " unknown_nets=" << stats.unknown_nets
            << " "
            << "malformed_rows=" << stats.malformed_rows
            << " self_res=" << stats.skipped_self_resistors
            << " unresolved_pin_nodes=" << stats.unresolved_pin_nodes
            << " pin_net_mismatches=" << stats.pin_net_mismatches
            << " missing_driver_nodes=" << stats.missing_driver_nodes
            << " missing_net_pins=" << stats.missing_net_pins
            << " fallback_net_pins=" << stats.fallback_net_pins
            << " repaired_edges=" << stats.repaired_edges
            << " loop_edges=" << stats.skipped_loop_edges
            << ". Regenerate/check my_dump_gr_rc rather than accepting repaired RC.";
        throw std::runtime_error(msg.str());
    }

    return graph;
}

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
    const RouteSegmentCacheMeta cache_meta = route_segment_cache_meta(file);
    const std::string cache_path = cache_enabled ? route_segment_cache_path(file) : std::string();
    const std::uint64_t cache_design_signature =
        route_segment_design_signature(gtdb, num_nets, num_pins);
    if (cache_enabled && cache_meta.source_size > 0) {
        HostRcGraph cached_graph;
        if (load_route_segment_cache(cache_path, cache_meta, num_nets, num_pins,
                                     missing_high_fanout_skip,
                                     cache_design_signature,
                                     cached_graph)) {
            logger.info("OpenROAD route-segment RC graph loaded from cache: %s",
                        cache_path.c_str());
            return cached_graph;
        }
    }

    std::ifstream input(file);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open OpenROAD route segment file: " + file);
    }
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
    int gpdb_direct_pins = 0;
    for (db::Net* dbnet : gtdb.rawdb.nets) {
        if (dbnet == nullptr) {
            continue;
        }
        for (db::Pin* dbpin : dbnet->pins) {
            if (dbpin == nullptr) {
                continue;
            }
            if (dbpin->gpdb_id >= 0 &&
                dbpin->gpdb_id < num_pins &&
                pin_id_to_dbpin[dbpin->gpdb_id] == nullptr) {
                pin_id_to_dbpin[dbpin->gpdb_id] = dbpin;
                gpdb_direct_pins++;
            }
        }
    }
    int unresolved_pins = 0;
    for (db::Pin* dbpin : pin_id_to_dbpin) {
        if (dbpin == nullptr) {
            unresolved_pins++;
        }
    }
    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=map_pins_by_gpdb elapsed=%.3f gpdb_direct=%d unresolved=%d total_timer_pins=%d\n",
                     seconds_since(build_start),
                     gpdb_direct_pins,
                     unresolved_pins,
                     num_pins);
        std::fflush(stderr);
    }

    int name_resolved_pins = 0;
    if (unresolved_pins > 0) {
        std::unordered_map<std::string, db::Pin*> pin_name_to_dbpin;
        pin_name_to_dbpin.reserve(static_cast<std::size_t>(unresolved_pins) * 4);
        auto add_dbpin_name_alias = [&](const std::string& name, db::Pin* dbpin) {
            if (name.empty() || dbpin == nullptr) {
                return;
            }
            auto add_alias = [&](const std::string& alias) {
                if (!alias.empty()) {
                    pin_name_to_dbpin.emplace(alias, dbpin);
                    const std::string normalized = normalized_spef_name(alias);
                    if (normalized != alias) {
                        pin_name_to_dbpin.emplace(normalized, dbpin);
                    }
                }
            };
            add_alias(name);
            add_alias(replace_char(name, '/', ':'));
            add_alias(replace_char(name, ':', '/'));
            add_alias(replace_last_char(name, '/', ':'));
            add_alias(replace_last_char(name, ':', '/'));
        };

        for (db::Net* dbnet : gtdb.rawdb.nets) {
            if (dbnet == nullptr) {
                continue;
            }
            for (db::Pin* dbpin : dbnet->pins) {
                if (dbpin == nullptr) {
                    continue;
                }
                if (dbpin->cell != nullptr && dbpin->type != nullptr) {
                    add_dbpin_name_alias(dbpin->cell->name() + ":" + dbpin->type->name(), dbpin);
                } else if (dbpin->iopin != nullptr) {
                    add_dbpin_name_alias(dbpin->iopin->name, dbpin);
                }
            }
        }
        profile_log("build_dbpin_alias_map");

        for (int pin_id = 0; pin_id < std::min(num_pins, static_cast<int>(gtdb.pin_names.size()));
             ++pin_id) {
            if (pin_id_to_dbpin[pin_id] != nullptr) {
                continue;
            }
            auto iter = pin_name_to_dbpin.find(gtdb.pin_names[pin_id]);
            if (iter == pin_name_to_dbpin.end()) {
                const std::string normalized = normalized_spef_name(gtdb.pin_names[pin_id]);
                if (normalized != gtdb.pin_names[pin_id]) {
                    iter = pin_name_to_dbpin.find(normalized);
                }
            }
            if (iter != pin_name_to_dbpin.end()) {
                pin_id_to_dbpin[pin_id] = iter->second;
                name_resolved_pins++;
            }
        }
    } else {
        profile_log("build_dbpin_alias_map_skipped");
    }
    logger.info("OpenROAD route-segment pin mapping: gpdb_direct=%d name_resolved=%d total_timer_pins=%d",
                gpdb_direct_pins, name_resolved_pins, num_pins);
    profile_log("resolve_timer_pins");

    auto resolve_net_token = [&](const char* begin, const char* end) {
        auto find_net = [&](std::string_view alias) {
            auto iter = net_name_to_index.find(alias);
            return iter == net_name_to_index.end() ? -1 : iter->second;
        };
        const std::string_view token(begin, static_cast<std::size_t>(end - begin));
        int net_idx = find_net(token);
        if (net_idx >= 0) {
            return net_idx;
        }
        const std::string name(begin, end);
        const std::string normalized = normalized_spef_name(name);
        if (normalized != name) {
            net_idx = find_net(std::string_view(normalized));
            if (net_idx >= 0) {
                return net_idx;
            }
        }
        const std::array<std::string, 4> aliases = {
            replace_char(name, '/', ':'),
            replace_char(name, ':', '/'),
            replace_last_char(name, '/', ':'),
            replace_last_char(name, ':', '/'),
        };
        for (const std::string& alias : aliases) {
            if (alias == name || alias == normalized) {
                continue;
            }
            net_idx = find_net(std::string_view(alias));
            if (net_idx >= 0) {
                return net_idx;
            }
        }
        return -1;
    };

    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=route_segment_options elapsed=%.3f missing_high_fanout_skip=%d\n",
                     seconds_since(build_start),
                     missing_high_fanout_skip);
        std::fflush(stderr);
    }

    auto resolve_layer_token = [&](const char* begin, const char* end) {
        const int trailing = trailing_integer_token(begin, end);
        if (trailing > 0) {
            return trailing;
        }
        std::string lower(begin, end);
        lower = lowercase_string(std::move(lower));
        auto iter = layer_name_to_level.find(lower);
        if (iter != layer_name_to_level.end()) {
            return iter->second;
        }
        return -1;
    };

    OpenroadRouteSegmentsBuildStats stats;
    constexpr int route_node_linear_scan_limit = 16;

    auto append_blank_node = [&](LocalSpefNetRc& local,
                                 int pin_id,
                                 const std::string& name,
                                 const OpenroadRoutePt& route_pt) {
        const int node_id = static_cast<int>(local.node2pin.size());
        local.node2pin.emplace_back(pin_id);
        if (keep_route_node_names) {
            local.node_names.emplace_back(name);
        }
        local.route_points.emplace_back(route_pt);
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
        return node_id;
    };

    using RouteNodeMap = std::unordered_map<OpenroadRoutePtKey, int, OpenroadRoutePtKeyHash>;
    std::vector<std::unique_ptr<LocalSpefNetRc>> local_nets(num_nets);
    std::vector<std::unique_ptr<RouteNodeMap>> route_node_maps(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);

    auto append_route_node = [&](int net_idx, const OpenroadRoutePtKey& key) {
        auto& local_ptr = local_nets[net_idx];
        if (!local_ptr) {
            local_ptr = std::make_unique<LocalSpefNetRc>();
        }
        LocalSpefNetRc& local = *local_ptr;
        auto& map_ptr = route_node_maps[net_idx];
        if (map_ptr) {
            auto iter = map_ptr->find(key);
            if (iter != map_ptr->end()) {
                return iter->second;
            }
        } else {
            for (int node = 0; node < static_cast<int>(local.route_points.size()); ++node) {
                if (route_point_matches(local.route_points[node], key)) {
                    return node;
                }
            }
            if (static_cast<int>(local.route_points.size()) >= route_node_linear_scan_limit) {
                map_ptr = std::make_unique<RouteNodeMap>();
                map_ptr->reserve(local.route_points.size() * 2 + 1);
                for (int node = 0; node < static_cast<int>(local.route_points.size()); ++node) {
                    const OpenroadRoutePt& pt = local.route_points[node];
                    if (pt.valid) {
                        map_ptr->emplace(OpenroadRoutePtKey{pt.x, pt.y, pt.layer}, node);
                    }
                }
            }
        }
        OpenroadRoutePt route_pt{key.x, key.y, key.layer, true};
        std::string name;
        if (keep_route_node_names) {
            name = std::to_string(key.x) + "," + std::to_string(key.y) + ",M" +
                   std::to_string(key.layer);
        }
        const int node_id = append_blank_node(local, -1, name, route_pt);
        if (map_ptr) {
            map_ptr->emplace(key, node_id);
        }
        return node_id;
    };

    auto add_edge = [](LocalSpefNetRc& local, int from, int to, float res) {
        local.edge_from.emplace_back(from);
        local.edge_to.emplace_back(to);
        local.edge_res.emplace_back(res);
    };

    std::string line;
    int current_net = -1;
    long long raw_lines = 0;
    while (std::getline(input, line)) {
        ++raw_lines;
        const char* line_begin = line.data();
        const char* line_end = line_begin + line.size();
        const char* first = skip_route_ws(line_begin, line_end);
        if (first == line_end || *first == '#') {
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
                continue;
            }
            if ((token_end - first) == 1 && *first == ')') {
                current_net = -1;
                continue;
            }
            current_net = resolve_net_token(first, token_end);
            if (current_net < 0 || current_net >= num_nets) {
                stats.unknown_nets++;
                current_net = -1;
                continue;
            }
            if (!parsed_net[current_net]) {
                parsed_net[current_net] = 1;
                stats.parsed_nets++;
            }
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
        if (current_net < 0 ||
            !parse_route_segment_row(line, x1, y1, layer1_begin, layer1_end,
                                     x2, y2, layer2_begin, layer2_end)) {
            stats.malformed_rows++;
            continue;
        }
        const int layer1 = resolve_layer_token(layer1_begin, layer1_end);
        const int layer2 = resolve_layer_token(layer2_begin, layer2_end);
        if (layer1 <= 0 || layer2 <= 0) {
            stats.unknown_layers++;
            continue;
        }

        const bool is_manhattan = (x1 == x2) || (y1 == y2);
        if (!is_manhattan) {
            stats.non_manhattan_segments++;
            continue;
        }

        const OpenroadRoutePtKey key1{x1, y1, layer1};
        const OpenroadRoutePtKey key2{x2, y2, layer2};
        const int from = append_route_node(current_net, key1);
        const int to = append_route_node(current_net, key2);
        if (from == to) {
            stats.skipped_self_segments++;
            continue;
        }

        long long dx = static_cast<long long>(x1) - static_cast<long long>(x2);
        long long dy = static_cast<long long>(y1) - static_cast<long long>(y2);
        if (dx < 0) dx = -dx;
        if (dy < 0) dy = -dy;
        const long long length_dbu = dx + dy;
        LocalSpefNetRc& local = *local_nets[current_net];
        if (length_dbu == 0) {
            const int lower_layer = std::min(layer1, layer2);
            add_edge(local, from, to, nangate45_via_res_ohm(lower_layer) / gtdb.res_unit);
            stats.via_segments++;
        } else if (layer1 == layer2) {
            const NangateLayerRc rc = nangate45_layer_rc(layer1);
            if (!(rc.res_ohm_per_um > 0.0f) && !(rc.cap_f_per_um > 0.0f)) {
                stats.unknown_layers++;
                continue;
            }
            const float length_um = static_cast<float>(length_dbu) / dbu_per_micron;
            const float edge_res = (rc.res_ohm_per_um * length_um) / gtdb.res_unit;
            const float cap = (rc.cap_f_per_um * length_um) / gtdb.cap_unit;
            add_edge(local, from, to, edge_res);
            add_attr_cap(local.node_cap, from, cap * 0.5f);
            add_attr_cap(local.node_cap, to, cap * 0.5f);
            stats.wire_segments++;
        } else {
            stats.non_manhattan_segments++;
            continue;
        }
        stats.segment_rows++;
    }
    if (profile) {
        std::fprintf(stderr,
                     "[ROUTE_SEG_PROFILE] phase=parse_segments elapsed=%.3f raw_lines=%lld segment_rows=%d parsed_nets=%d route_nodes_pending=%zu\n",
                     seconds_since(build_start),
                     raw_lines,
                     stats.segment_rows,
                     stats.parsed_nets,
                     static_cast<std::size_t>(stats.parsed_nets));
        std::fflush(stderr);
    }

    auto append_pin_node = [&](LocalSpefNetRc& local, int pin_id) {
        OpenroadRoutePt route_pt;
        std::string name;
        if (keep_route_node_names &&
            pin_id >= 0 &&
            pin_id < static_cast<int>(gtdb.pin_names.size())) {
            name = gtdb.pin_names[pin_id];
        }
        return append_blank_node(local, pin_id, name, route_pt);
    };

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

    auto pin_openroad_route_loc = [&](int pin_id, OpenroadPinRouteLoc& loc) {
        loc = OpenroadPinRouteLoc{};
        if (pin_id < 0 || pin_id >= static_cast<int>(pin_id_to_dbpin.size()) ||
            pin_id_to_dbpin[pin_id] == nullptr) {
            return false;
        }
        db::Pin* pin = pin_id_to_dbpin[pin_id];
        if (pin->type == nullptr) {
            return false;
        }

        std::map<int, std::vector<std::array<int, 4>>> boxes_by_layer;
        auto add_box = [&](int routing_layer, int lx, int ly, int hx, int hy) {
            loc.pin_x = lx;
            loc.pin_y = ly;
            loc.pin_layer = routing_layer;
            boxes_by_layer[routing_layer].push_back({lx, ly, hx, hy});
        };

        if (pin->cell != nullptr) {
            db::Cell* cell = pin->cell;
            db::CellType* ctype = cell->ctype();
            const int dx = cell->lx() + (ctype == nullptr ? 0 : ctype->originX());
            const int dy = cell->ly() + (ctype == nullptr ? 0 : ctype->originY());
            for (const db::Geometry& shape : pin->type->shapes) {
                if (shape.layer.rIndex < 0) {
                    continue;
                }
                int lx = 0;
                int ly = 0;
                int hx = 0;
                int hy = 0;
                std::tie(lx, ly, hx, hy) =
                    orient_box_for_cell(ctype, cell->orient(), shape.lx, shape.ly, shape.hx, shape.hy);
                add_box(shape.layer.rIndex + 1, dx + lx, dy + ly, dx + hx, dy + hy);
            }
        } else if (pin->iopin != nullptr) {
            db::IOPin* iopin = pin->iopin;
            for (const db::Geometry& shape : pin->type->shapes) {
                if (shape.layer.rIndex < 0) {
                    continue;
                }
                int lx = 0;
                int ly = 0;
                int hx = 0;
                int hy = 0;
                std::tie(lx, ly, hx, hy) =
                    orient_box_for_iopin(iopin->orient(), shape.lx, shape.ly, shape.hx, shape.hy);
                add_box(shape.layer.rIndex + 1,
                        iopin->x + lx,
                        iopin->y + ly,
                        iopin->x + hx,
                        iopin->y + hy);
            }
        }

        if (boxes_by_layer.empty()) {
            return false;
        }

        const int conn_layer = boxes_by_layer.rbegin()->first;
        loc.conn_layer = conn_layer;

        const db::Layer* conn_db_layer = nullptr;
        for (const db::Layer& layer : gtdb.rawdb.layers) {
            if (layer.rIndex + 1 == conn_layer) {
                conn_db_layer = &layer;
                break;
            }
        }
        bool adjust_single_track = false;
        bool adjust_horizontal = false;
        int adjusted_track = 0;
        if (conn_db_layer != nullptr && !conn_db_layer->tracks.empty()) {
            const db::Track& track = conn_db_layer->tracks.front();
            if (track.step > 0) {
                int min_coord = std::numeric_limits<int>::max();
                int max_coord = std::numeric_limits<int>::min();
                const bool horizontal = conn_db_layer->direction == 'h';
                for (const auto& box : boxes_by_layer[conn_layer]) {
                    min_coord = std::min(min_coord, horizontal ? box[1] : box[0]);
                    max_coord = std::max(max_coord, horizontal ? box[3] : box[2]);
                }
                if (min_coord <= max_coord &&
                    static_cast<float>(max_coord - min_coord) /
                            static_cast<float>(track.step) <=
                        3.0f) {
                    const int nearest_track =
                        static_cast<int>(std::floor(
                            (static_cast<float>(max_coord - track.start)) /
                            static_cast<float>(track.step))) *
                            static_cast<int>(track.step) +
                        track.start;
                    const int nearest_track2 =
                        static_cast<int>(std::floor(
                            (static_cast<float>(max_coord - track.start)) /
                                static_cast<float>(track.step) -
                            1.0f)) *
                            static_cast<int>(track.step) +
                        track.start;
                    const bool first_inside =
                        nearest_track >= min_coord && nearest_track <= max_coord;
                    const bool second_inside =
                        nearest_track2 >= min_coord && nearest_track2 <= max_coord;
                    if (!(first_inside && second_inside)) {
                        if (nearest_track > min_coord && nearest_track < max_coord) {
                            adjust_single_track = true;
                            adjust_horizontal = horizontal;
                            adjusted_track = nearest_track;
                        } else if (nearest_track2 > min_coord && nearest_track2 < max_coord) {
                            adjust_single_track = true;
                            adjust_horizontal = horizontal;
                            adjusted_track = nearest_track2;
                        }
                    }
                }
            }
        }

        std::map<std::pair<int, int>, int> grid_votes;
        std::vector<std::pair<int, int>> grid_order;
        for (const auto& box : boxes_by_layer[conn_layer]) {
            int cx = box[0] + (box[2] - box[0]) / 2;
            int cy = box[1] + (box[3] - box[1]) / 2;
            if (adjust_single_track) {
                if (adjust_horizontal) {
                    cy = adjusted_track;
                } else {
                    cx = adjusted_track;
                }
            }
            const auto grid = openroad_position_on_inferred_grid(gtdb.rawdb, openroad_grid, cx, cy);
            if (grid_votes.emplace(grid, 0).second) {
                grid_order.emplace_back(grid);
            }
            grid_votes[grid]++;
        }

        int best_votes = -1;
        for (const auto& grid : grid_order) {
            const int votes = grid_votes[grid];
            if (votes > best_votes) {
                best_votes = votes;
                loc.grid_x = grid.first;
                loc.grid_y = grid.second;
            }
        }
        for (const auto& box : boxes_by_layer[conn_layer]) {
            int cx = box[0] + (box[2] - box[0]) / 2;
            int cy = box[1] + (box[3] - box[1]) / 2;
            if (adjust_single_track) {
                if (adjust_horizontal) {
                    cy = adjusted_track;
                } else {
                    cx = adjusted_track;
                }
            }
            const auto grid = openroad_position_on_inferred_grid(gtdb.rawdb, openroad_grid, cx, cy);
            if (grid.first == loc.grid_x && grid.second == loc.grid_y) {
                loc.grid_src_x = cx;
                loc.grid_src_y = cy;
                break;
            }
        }

        loc.valid = true;
        return true;
    };

    auto reorder_root = [](LocalSpefNetRc& local, int root_node) {
        if (root_node <= 0 || root_node >= static_cast<int>(local.node2pin.size())) {
            return;
        }
        std::swap(local.node2pin[0], local.node2pin[root_node]);
        if (root_node < static_cast<int>(local.node_names.size())) {
            std::swap(local.node_names[0], local.node_names[root_node]);
        }
        if (root_node < static_cast<int>(local.route_points.size())) {
            std::swap(local.route_points[0], local.route_points[root_node]);
        }
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            std::swap(local.node_cap[attr], local.node_cap[root_node * NUM_ATTR + attr]);
        }
        for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            if (local.edge_from[edge] == 0) {
                local.edge_from[edge] = root_node;
            } else if (local.edge_from[edge] == root_node) {
                local.edge_from[edge] = 0;
            }
            if (local.edge_to[edge] == 0) {
                local.edge_to[edge] = root_node;
            } else if (local.edge_to[edge] == root_node) {
                local.edge_to[edge] = 0;
            }
        }
    };

    HostRcGraph graph;
    graph.includes_pin_caps.assign(num_nets, 0);
    graph.net2node_start.emplace_back(0);
    graph.net2edge_start.emplace_back(0);

    double final_pinloc_seconds = 0.0;
    double final_attach_seconds = 0.0;
    double final_reorder_seconds = 0.0;
    double final_repair_seconds = 0.0;
    double final_prune_seconds = 0.0;
    double final_append_seconds = 0.0;
    int repair_adjacency_nets = 0;
    int repair_scan_nets = 0;
    long long repair_node_edge_product_max = 0;
    int progress_interval = 0;
    if (profile) {
        progress_interval = 10000;
        if (const char* interval_env = std::getenv("GPUTIMER_ROUTE_SEG_PROFILE_INTERVAL")) {
            const int parsed_interval = std::atoi(interval_env);
            if (parsed_interval > 0) {
                progress_interval = parsed_interval;
            }
        }
    }

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        auto phase_start = std::chrono::steady_clock::now();
        LocalSpefNetRc local;
        if (local_nets[net_idx]) {
            local = std::move(*local_nets[net_idx]);
        }
        const int pin_begin = flat_net2pin_start_map[net_idx];
        const int pin_end = flat_net2pin_start_map[net_idx + 1];
        const int fanout = pin_end - pin_begin;
        const int driver_pin = pin_begin < pin_end ? flat_net2pin_map[pin_begin] : -1;

        if (profile &&
            (net_idx % progress_interval == 0 ||
             fanout > 10000 ||
             local.node2pin.size() > 10000 ||
             local.edge_from.size() > 10000)) {
            std::fprintf(stderr,
                         "[ROUTE_SEG_PROFILE] phase=finalize_net_start elapsed=%.3f net=%d/%d name=%s parsed=%d fanout=%d local_nodes=%zu local_edges=%zu\n",
                         seconds_since(build_start),
                         net_idx,
                         num_nets,
                         (net_idx >= 0 && net_idx < static_cast<int>(gtdb.net_names.size()))
                             ? gtdb.net_names[net_idx].c_str()
                             : "<bad>",
                         parsed_net[net_idx] ? 1 : 0,
                         fanout,
                         local.node2pin.size(),
                         local.edge_from.size());
            std::fflush(stderr);
        }

        if (!parsed_net[net_idx]) {
            stats.missing_nets++;
        }

        if (!parsed_net[net_idx] && fanout <= 1) {
            stats.skipped_missing_unconnected_nets++;
            stats.skipped_missing_unconnected_pins += std::max(fanout, 0);
            graph.net2node_start.emplace_back(graph.num_nodes);
            graph.net2edge_start.emplace_back(graph.num_edges);
            final_append_seconds += seconds_since(phase_start);
            if (progress_interval > 0 && (net_idx + 1) % progress_interval == 0) {
                std::fprintf(stderr,
                             "[ROUTE_SEG_PROFILE] phase=finalize_progress elapsed=%.3f nets=%d/%d nodes=%d edges=%d pinloc=%.3f attach=%.3f reorder=%.3f repair=%.3f prune=%.3f append=%.3f adj_nets=%d scan_nets=%d max_node_edge_product=%lld skipped_missing_high_fanout_nets=%d skipped_missing_high_fanout_pins=%lld skipped_missing_unconnected_nets=%d skipped_missing_unconnected_pins=%lld\n",
                             seconds_since(build_start),
                             net_idx + 1,
                             num_nets,
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
            continue;
        }

        if (!parsed_net[net_idx] &&
            missing_high_fanout_skip > 0 &&
            fanout > missing_high_fanout_skip) {
            stats.skipped_missing_high_fanout_nets++;
            stats.skipped_missing_high_fanout_pins += fanout;
            if (driver_pin >= 0) {
                const int driver_node = append_pin_node(local, driver_pin);
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
            for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                graph.edge_from.emplace_back(graph.num_nodes + local.edge_from[edge]);
                graph.edge_to.emplace_back(graph.num_nodes + local.edge_to[edge]);
                graph.edge_res.emplace_back(local.edge_res[edge]);
                graph.num_edges++;
            }
            for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
                graph.node2pin.emplace_back(local.node2pin[node]);
                if (keep_route_node_names) {
                    graph.node_names.emplace_back(local.node_names[node]);
                }
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    graph.node_cap.emplace_back(local.node_cap[node * NUM_ATTR + attr]);
                }
            }
            graph.num_nodes += static_cast<int>(local.node2pin.size());
            graph.net2node_start.emplace_back(graph.num_nodes);
            graph.net2edge_start.emplace_back(graph.num_edges);
            final_append_seconds += seconds_since(phase_start);
            if (progress_interval > 0 && (net_idx + 1) % progress_interval == 0) {
                std::fprintf(stderr,
                             "[ROUTE_SEG_PROFILE] phase=finalize_progress elapsed=%.3f nets=%d/%d nodes=%d edges=%d pinloc=%.3f attach=%.3f reorder=%.3f repair=%.3f prune=%.3f append=%.3f adj_nets=%d scan_nets=%d max_node_edge_product=%lld skipped_missing_high_fanout_nets=%d skipped_missing_high_fanout_pins=%lld\n",
                             seconds_since(build_start),
                             net_idx + 1,
                             num_nets,
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
                             stats.skipped_missing_high_fanout_pins);
                std::fflush(stderr);
            }
            continue;
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
            pin_openroad_route_loc(pin_id, loc);
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
        final_pinloc_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        if (local.node2pin.empty() && driver_pin >= 0) {
            append_pin_node(local, driver_pin);
        }

        std::unordered_set<int> present_pins;
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

            const int pin_node = append_pin_node(local, pin_id);
            const OpenroadPinRouteLoc& pin_loc = pin_route_locs[pin_pos - pin_begin];
            const bool have_pin_loc = pin_loc.valid;

            int route_node = -1;
            if (have_pin_loc && min_route_layer > 0) {
                RouteNodeMap* route_map = route_node_maps[net_idx].get();
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
                if (parsed_net[net_idx]) {
                    stats.missing_net_pins++;
                } else {
                    stats.fallback_net_pins++;
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
                    stats.pin_stub_edges++;
                } else if (parsed_net[net_idx]) {
                    stats.missing_net_pins++;
                } else {
                    stats.fallback_net_pins++;
                }
                add_edge(local, pin_node, route_node, edge_res);
            }

        }
        final_attach_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        int driver_node = -1;
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            if (local.node2pin[node] == driver_pin) {
                driver_node = node;
                break;
            }
        }
        if (driver_pin >= 0 && driver_node < 0) {
            driver_node = append_pin_node(local, driver_pin);
            stats.missing_driver_nodes++;
        }
        reorder_root(local, driver_node);
        final_reorder_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);

            const long long node_edge_product =
                static_cast<long long>(local.node2pin.size()) *
                static_cast<long long>(local.edge_from.size());
            repair_node_edge_product_max = std::max(repair_node_edge_product_max,
                                                    node_edge_product);

            if (node_edge_product > 4096) {
                ++repair_adjacency_nets;
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
                ++repair_scan_nets;
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
                    stats.repaired_edges++;
                }
            }
        }
        final_repair_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        const int skipped_loop_edges = prune_to_rooted_tree(local);
        if (skipped_loop_edges > 0) {
            stats.skipped_loop_edges += skipped_loop_edges;
        }
        final_prune_seconds += seconds_since(phase_start);
        phase_start = std::chrono::steady_clock::now();

        for (std::size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            graph.edge_from.emplace_back(graph.num_nodes + local.edge_from[edge]);
            graph.edge_to.emplace_back(graph.num_nodes + local.edge_to[edge]);
            graph.edge_res.emplace_back(local.edge_res[edge]);
            graph.num_edges++;
        }
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            graph.node2pin.emplace_back(local.node2pin[node]);
            if (keep_route_node_names) {
                graph.node_names.emplace_back(local.node_names[node]);
            }
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                graph.node_cap.emplace_back(local.node_cap[node * NUM_ATTR + attr]);
            }
        }
        graph.num_nodes += static_cast<int>(local.node2pin.size());
        graph.net2node_start.emplace_back(graph.num_nodes);
        graph.net2edge_start.emplace_back(graph.num_edges);
        final_append_seconds += seconds_since(phase_start);

        if (progress_interval > 0 && (net_idx + 1) % progress_interval == 0) {
            std::fprintf(stderr,
                         "[ROUTE_SEG_PROFILE] phase=finalize_progress elapsed=%.3f nets=%d/%d nodes=%d edges=%d pinloc=%.3f attach=%.3f reorder=%.3f repair=%.3f prune=%.3f append=%.3f adj_nets=%d scan_nets=%d max_node_edge_product=%lld\n",
                         seconds_since(build_start),
                         net_idx + 1,
                         num_nets,
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
                         repair_node_edge_product_max);
            std::fflush(stderr);
        }
    }

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

void GPUTimer::debug_dump_openroad_gr_rc_net(const std::string& file,
                                             const std::string& net_name) {
    HostRcGraph graph = build_openroad_gr_rc(file);
    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    int net_idx = -1;
    auto iter = net_name_to_index.find(net_name);
    if (iter != net_name_to_index.end()) {
        net_idx = iter->second;
    } else {
        std::string normalized = net_name;
        validate_token(normalized);
        iter = net_name_to_index.find(normalized);
        if (iter != net_name_to_index.end()) {
            net_idx = iter->second;
        }
    }
    if (net_idx < 0) {
        logger.warning("OpenROAD GR RC dump: net %s not found", net_name.c_str());
        return;
    }

    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[GR RC DUMP] file=%s net=%s id=%d nodes=%d edges=%d includes_pin_caps=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           file.c_str(), gtdb.net_names[net_idx].c_str(), net_idx, nend - nst,
           eend - est, graph.includes_pin_caps[net_idx] ? 1 : 0,
           graph.repaired_edges, graph.skipped_loop_edges);
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[GR RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node, node, pin, graph.node_names[node].c_str(),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[GR RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e\n",
               edge - est, edge, graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst, graph.edge_res[edge]);
    }
    fflush(stdout);
}

void GPUTimer::debug_dump_openroad_route_segments_rc_net(
    const std::string& file,
    const std::string& net_name)
{
    HostRcGraph graph = build_openroad_route_segments_rc(file);
    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_gr_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    int net_idx = -1;
    auto iter = net_name_to_index.find(net_name);
    if (iter != net_name_to_index.end()) {
        net_idx = iter->second;
    } else {
        std::string normalized = net_name;
        validate_token(normalized);
        iter = net_name_to_index.find(normalized);
        if (iter != net_name_to_index.end()) {
            net_idx = iter->second;
        }
    }
    if (net_idx < 0) {
        logger.warning("OpenROAD route segment RC: net %s not found", net_name.c_str());
        return;
    }

    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[ROUTE SEG RC DUMP] file=%s net=%s id=%d nodes=%d edges=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           file.c_str(),
           gtdb.net_names[net_idx].c_str(),
           net_idx,
           nend - nst,
           eend - est,
           graph.repaired_edges,
           graph.skipped_loop_edges);
    auto node_name = [&](int global_node) -> const char* {
        if (global_node >= 0 &&
            global_node < static_cast<int>(graph.node_names.size())) {
            return graph.node_names[global_node].c_str();
        }
        return "";
    };
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[ROUTE SEG RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node,
               node,
               pin,
               node_name(node),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[ROUTE SEG RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e from_name=%s to_name=%s\n",
               edge - est,
               edge,
               graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst,
               graph.edge_res[edge],
               node_name(graph.edge_from[edge]),
               node_name(graph.edge_to[edge]));
    }
    fflush(stdout);
}

void GPUTimer::debug_compare_openroad_route_segments_rc(
    const std::string& gr_rc_file,
    const std::string& route_segments_file,
    int top_n)
{
    HostRcGraph gr_graph = build_openroad_gr_rc(gr_rc_file);
    HostRcGraph route_graph = build_openroad_route_segments_rc(route_segments_file);

    struct NetDiff {
        int net_idx = -1;
        int gr_nodes = 0;
        int route_nodes = 0;
        int gr_edges = 0;
        int route_edges = 0;
        double edge_abs = 0.0;
        double cap_abs = 0.0;
        float max_edge_abs = 0.0f;
        float max_cap_abs = 0.0f;
        int max_edge = -1;
        int max_cap_node = -1;
        int max_cap_attr = -1;
        bool shape_mismatch = false;
    };

    struct SemanticEdgeValue {
        double res = 0.0;
        int count = 0;
    };

    struct SemanticNetDiff {
        int net_idx = -1;
        int gr_nodes = 0;
        int route_nodes = 0;
        int gr_edges = 0;
        int route_edges = 0;
        int missing_nodes = 0;
        int missing_edges = 0;
        int edge_count_mismatches = 0;
        int fallback_keys = 0;
        double edge_abs = 0.0;
        double cap_abs = 0.0;
        double max_edge_abs = 0.0;
        double max_cap_abs = 0.0;
        double max_edge_gr_res = 0.0;
        double max_edge_route_res = 0.0;
        std::string max_edge_key;
        std::string max_cap_key;
    };

    auto semantic_node_key = [&](const HostRcGraph& graph,
                                 int global_node,
                                 int net_node_start,
                                 int& fallback_keys) {
        const int pin_id = graph.node2pin[global_node];
        if (pin_id >= 0) {
            std::string key = std::string("P:") + std::to_string(pin_id);
            if (pin_id < static_cast<int>(gtdb.pin_names.size())) {
                key += ":" + gtdb.pin_names[pin_id];
            }
            return key;
        }
        if (global_node >= 0 && global_node < static_cast<int>(graph.node_names.size()) &&
            !graph.node_names[global_node].empty()) {
            return std::string("R:") + graph.node_names[global_node];
        }
        fallback_keys++;
        return std::string("L:") + std::to_string(global_node - net_node_start);
    };

    auto semantic_edge_key = [](const std::string& lhs, const std::string& rhs) {
        if (lhs < rhs) {
            return lhs + "\t" + rhs;
        }
        return rhs + "\t" + lhs;
    };

    auto build_semantic_node_caps =
        [&](const HostRcGraph& graph,
            int nst,
            int nend,
            int& fallback_keys) {
            std::unordered_map<std::string, std::array<double, NUM_ATTR>> caps;
            for (int node = nst; node < nend; ++node) {
                const std::string key = semantic_node_key(graph, node, nst, fallback_keys);
                auto& cap = caps[key];
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    cap[attr] += graph.node_cap[node * NUM_ATTR + attr];
                }
            }
            return caps;
        };

    auto build_semantic_edges =
        [&](const HostRcGraph& graph,
            int nst,
            int est,
            int eend,
            int& fallback_keys) {
            std::unordered_map<std::string, SemanticEdgeValue> edges;
            for (int edge = est; edge < eend; ++edge) {
                const std::string from =
                    semantic_node_key(graph, graph.edge_from[edge], nst, fallback_keys);
                const std::string to =
                    semantic_node_key(graph, graph.edge_to[edge], nst, fallback_keys);
                auto& value = edges[semantic_edge_key(from, to)];
                value.res += graph.edge_res[edge];
                value.count++;
            }
            return edges;
        };

    std::vector<NetDiff> diffs;
    std::vector<SemanticNetDiff> semantic_diffs;
    diffs.reserve(num_nets);
    semantic_diffs.reserve(num_nets);
    double total_edge_abs = 0.0;
    double total_cap_abs = 0.0;
    double semantic_total_edge_abs = 0.0;
    double semantic_total_cap_abs = 0.0;
    int shape_mismatch_nets = 0;
    int semantic_shape_mismatch_nets = 0;
    int semantic_missing_nodes = 0;
    int semantic_missing_edges = 0;
    int semantic_edge_count_mismatches = 0;
    int semantic_fallback_keys = 0;

    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        const int gr_nst = gr_graph.net2node_start[net_idx];
        const int gr_nend = gr_graph.net2node_start[net_idx + 1];
        const int gr_est = gr_graph.net2edge_start[net_idx];
        const int gr_eend = gr_graph.net2edge_start[net_idx + 1];
        const int route_nst = route_graph.net2node_start[net_idx];
        const int route_nend = route_graph.net2node_start[net_idx + 1];
        const int route_est = route_graph.net2edge_start[net_idx];
        const int route_eend = route_graph.net2edge_start[net_idx + 1];

        NetDiff diff;
        diff.net_idx = net_idx;
        diff.gr_nodes = gr_nend - gr_nst;
        diff.route_nodes = route_nend - route_nst;
        diff.gr_edges = gr_eend - gr_est;
        diff.route_edges = route_eend - route_est;
        diff.shape_mismatch = diff.gr_nodes != diff.route_nodes ||
                              diff.gr_edges != diff.route_edges;
        if (diff.shape_mismatch) {
            shape_mismatch_nets++;
        }

        const int edge_count = std::min(diff.gr_edges, diff.route_edges);
        for (int edge = 0; edge < edge_count; ++edge) {
            const float delta = std::fabs(gr_graph.edge_res[gr_est + edge] -
                                          route_graph.edge_res[route_est + edge]);
            diff.edge_abs += delta;
            if (delta > diff.max_edge_abs) {
                diff.max_edge_abs = delta;
                diff.max_edge = edge;
            }
        }

        const int node_count = std::min(diff.gr_nodes, diff.route_nodes);
        for (int node = 0; node < node_count; ++node) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const float delta = std::fabs(
                    gr_graph.node_cap[(gr_nst + node) * NUM_ATTR + attr] -
                    route_graph.node_cap[(route_nst + node) * NUM_ATTR + attr]);
                diff.cap_abs += delta;
                if (delta > diff.max_cap_abs) {
                    diff.max_cap_abs = delta;
                    diff.max_cap_node = node;
                    diff.max_cap_attr = attr;
                }
            }
        }

        total_edge_abs += diff.edge_abs;
        total_cap_abs += diff.cap_abs;
        if (diff.shape_mismatch || diff.edge_abs > 0.0 || diff.cap_abs > 0.0) {
            diffs.emplace_back(diff);
        }

        SemanticNetDiff sem_diff;
        sem_diff.net_idx = net_idx;
        sem_diff.gr_nodes = diff.gr_nodes;
        sem_diff.route_nodes = diff.route_nodes;
        sem_diff.gr_edges = diff.gr_edges;
        sem_diff.route_edges = diff.route_edges;

        int net_fallback_keys = 0;
        auto gr_node_caps = build_semantic_node_caps(gr_graph, gr_nst, gr_nend, net_fallback_keys);
        auto route_node_caps =
            build_semantic_node_caps(route_graph, route_nst, route_nend, net_fallback_keys);
        auto gr_edges =
            build_semantic_edges(gr_graph, gr_nst, gr_est, gr_eend, net_fallback_keys);
        auto route_edges =
            build_semantic_edges(route_graph, route_nst, route_est, route_eend, net_fallback_keys);
        sem_diff.fallback_keys = net_fallback_keys;

        std::unordered_set<std::string> node_keys;
        node_keys.reserve(gr_node_caps.size() + route_node_caps.size());
        for (const auto& item : gr_node_caps) {
            node_keys.insert(item.first);
        }
        for (const auto& item : route_node_caps) {
            node_keys.insert(item.first);
        }
        for (const std::string& key : node_keys) {
            const auto gr_iter = gr_node_caps.find(key);
            const auto route_iter = route_node_caps.find(key);
            if (gr_iter == gr_node_caps.end() || route_iter == route_node_caps.end()) {
                sem_diff.missing_nodes++;
            }
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const double gr_cap =
                    gr_iter == gr_node_caps.end() ? 0.0 : gr_iter->second[attr];
                const double route_cap =
                    route_iter == route_node_caps.end() ? 0.0 : route_iter->second[attr];
                const double delta = std::fabs(gr_cap - route_cap);
                sem_diff.cap_abs += delta;
                if (delta > sem_diff.max_cap_abs) {
                    sem_diff.max_cap_abs = delta;
                    sem_diff.max_cap_key = key + "/attr" + std::to_string(attr);
                }
            }
        }

        std::unordered_set<std::string> edge_keys;
        edge_keys.reserve(gr_edges.size() + route_edges.size());
        for (const auto& item : gr_edges) {
            edge_keys.insert(item.first);
        }
        for (const auto& item : route_edges) {
            edge_keys.insert(item.first);
        }
        for (const std::string& key : edge_keys) {
            const auto gr_iter = gr_edges.find(key);
            const auto route_iter = route_edges.find(key);
            if (gr_iter == gr_edges.end() || route_iter == route_edges.end()) {
                sem_diff.missing_edges++;
            }
            const double gr_res = gr_iter == gr_edges.end() ? 0.0 : gr_iter->second.res;
            const double route_res =
                route_iter == route_edges.end() ? 0.0 : route_iter->second.res;
            const int gr_count = gr_iter == gr_edges.end() ? 0 : gr_iter->second.count;
            const int route_count = route_iter == route_edges.end() ? 0 : route_iter->second.count;
            if (gr_count != route_count) {
                sem_diff.edge_count_mismatches++;
            }
            const double delta = std::fabs(gr_res - route_res);
            sem_diff.edge_abs += delta;
            if (delta > sem_diff.max_edge_abs) {
                sem_diff.max_edge_abs = delta;
                sem_diff.max_edge_gr_res = gr_res;
                sem_diff.max_edge_route_res = route_res;
                sem_diff.max_edge_key = key;
            }
        }

        semantic_total_edge_abs += sem_diff.edge_abs;
        semantic_total_cap_abs += sem_diff.cap_abs;
        semantic_missing_nodes += sem_diff.missing_nodes;
        semantic_missing_edges += sem_diff.missing_edges;
        semantic_edge_count_mismatches += sem_diff.edge_count_mismatches;
        semantic_fallback_keys += sem_diff.fallback_keys;
        const bool semantic_shape_mismatch =
            sem_diff.missing_nodes > 0 ||
            sem_diff.missing_edges > 0 ||
            sem_diff.edge_count_mismatches > 0 ||
            sem_diff.fallback_keys > 0;
        if (semantic_shape_mismatch) {
            semantic_shape_mismatch_nets++;
        }
        if (semantic_shape_mismatch ||
            sem_diff.edge_abs > 0.0 ||
            sem_diff.cap_abs > 0.0) {
            semantic_diffs.emplace_back(std::move(sem_diff));
        }
    }

    std::sort(diffs.begin(), diffs.end(), [](const NetDiff& lhs, const NetDiff& rhs) {
        if (lhs.shape_mismatch != rhs.shape_mismatch) {
            return lhs.shape_mismatch > rhs.shape_mismatch;
        }
        const double lhs_score = lhs.edge_abs + lhs.cap_abs;
        const double rhs_score = rhs.edge_abs + rhs.cap_abs;
        return lhs_score > rhs_score;
    });

    std::sort(semantic_diffs.begin(),
              semantic_diffs.end(),
              [](const SemanticNetDiff& lhs, const SemanticNetDiff& rhs) {
                  const bool lhs_shape = lhs.missing_nodes > 0 ||
                                         lhs.missing_edges > 0 ||
                                         lhs.edge_count_mismatches > 0 ||
                                         lhs.fallback_keys > 0;
                  const bool rhs_shape = rhs.missing_nodes > 0 ||
                                         rhs.missing_edges > 0 ||
                                         rhs.edge_count_mismatches > 0 ||
                                         rhs.fallback_keys > 0;
                  if (lhs_shape != rhs_shape) {
                      return lhs_shape > rhs_shape;
                  }
                  const double lhs_score = lhs.edge_abs + lhs.cap_abs;
                  const double rhs_score = rhs.edge_abs + rhs.cap_abs;
                  return lhs_score > rhs_score;
              });

    printf("[GR RC COMPARE] gr_rc=%s\n", gr_rc_file.c_str());
    printf("[GR RC COMPARE] route_segments=%s\n", route_segments_file.c_str());
    printf("[GR RC SEMCOMPARE] shape_mismatch_nets=%d missing_nodes=%d missing_edges=%d edge_count_mismatches=%d fallback_keys=%d total_edge_abs=%.9e total_cap_abs=%.9e changed_nets=%zu\n",
           semantic_shape_mismatch_nets,
           semantic_missing_nodes,
           semantic_missing_edges,
           semantic_edge_count_mismatches,
           semantic_fallback_keys,
           semantic_total_edge_abs,
           semantic_total_cap_abs,
           semantic_diffs.size());
    const int semantic_limit =
        std::max(0, std::min(top_n, static_cast<int>(semantic_diffs.size())));
    for (int i = 0; i < semantic_limit; ++i) {
        const SemanticNetDiff& diff = semantic_diffs[i];
        printf("[GR RC SEMCOMPARE] rank=%d net=%s id=%d nodes=%d/%d edges=%d/%d missing_nodes=%d missing_edges=%d edge_count_mismatches=%d fallback_keys=%d edge_abs=%.9e cap_abs=%.9e max_edge_abs=%.9e max_edge_gr=%.9e max_edge_route=%.9e edge_key=%s max_cap_abs=%.9e cap_key=%s\n",
               i + 1,
               gtdb.net_names[diff.net_idx].c_str(),
               diff.net_idx,
               diff.gr_nodes,
               diff.route_nodes,
               diff.gr_edges,
               diff.route_edges,
               diff.missing_nodes,
               diff.missing_edges,
               diff.edge_count_mismatches,
               diff.fallback_keys,
               diff.edge_abs,
               diff.cap_abs,
               diff.max_edge_abs,
               diff.max_edge_gr_res,
               diff.max_edge_route_res,
               diff.max_edge_key.c_str(),
               diff.max_cap_abs,
               diff.max_cap_key.c_str());
    }
    printf("[GR RC COMPARE] gr nodes=%d edges=%d route nodes=%d edges=%d shape_mismatch_nets=%d total_edge_abs=%.9e total_cap_abs=%.9e changed_nets=%zu\n",
           gr_graph.num_nodes,
           gr_graph.num_edges,
           route_graph.num_nodes,
           route_graph.num_edges,
           shape_mismatch_nets,
           total_edge_abs,
           total_cap_abs,
           diffs.size());

    const int limit = std::max(0, std::min(top_n, static_cast<int>(diffs.size())));
    for (int i = 0; i < limit; ++i) {
        const NetDiff& diff = diffs[i];
        printf("[GR RC COMPARE] rank=%d net=%s id=%d nodes=%d/%d edges=%d/%d shape_mismatch=%d edge_abs=%.9e cap_abs=%.9e max_edge_abs=%.9e edge=%d max_cap_abs=%.9e cap_node=%d cap_attr=%d\n",
               i + 1,
               gtdb.net_names[diff.net_idx].c_str(),
               diff.net_idx,
               diff.gr_nodes,
               diff.route_nodes,
               diff.gr_edges,
               diff.route_edges,
               diff.shape_mismatch ? 1 : 0,
               diff.edge_abs,
               diff.cap_abs,
               diff.max_edge_abs,
               diff.max_edge,
               diff.max_cap_abs,
               diff.max_cap_node,
               diff.max_cap_attr);
    }
    fflush(stdout);
}


}  // namespace gt
