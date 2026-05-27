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
