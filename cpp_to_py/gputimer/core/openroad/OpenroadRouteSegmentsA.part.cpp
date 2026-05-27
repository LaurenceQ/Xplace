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
