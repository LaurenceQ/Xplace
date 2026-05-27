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
