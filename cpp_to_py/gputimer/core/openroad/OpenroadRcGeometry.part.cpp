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
