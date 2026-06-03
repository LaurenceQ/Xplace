#include "gputimer/core/openroad/OpenroadRcInternal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gt {
namespace openroad_rc {

namespace {

constexpr int kInlinePinBoxCapacity = 8;
using PinBox = std::array<int, 4>;
using PinGrid = std::pair<int, int>;

struct PinBoxList {
    std::array<PinBox, kInlinePinBoxCapacity> inline_boxes{};
    int inline_count = 0;
    std::vector<PinBox> overflow;

    void clear()
    {
        inline_count = 0;
        overflow.clear();
    }

    bool empty() const
    {
        return inline_count == 0 && overflow.empty();
    }

    int size() const
    {
        return overflow.empty() ? inline_count : static_cast<int>(overflow.size());
    }

    const PinBox& front() const
    {
        return overflow.empty() ? inline_boxes[0] : overflow.front();
    }

    void push(const PinBox& box)
    {
        if (overflow.empty() && inline_count < kInlinePinBoxCapacity) {
            inline_boxes[inline_count++] = box;
            return;
        }
        if (overflow.empty()) {
            overflow.reserve(kInlinePinBoxCapacity * 2);
            for (int i = 0; i < inline_count; ++i) {
                overflow.emplace_back(inline_boxes[i]);
            }
            inline_count = 0;
        }
        overflow.emplace_back(box);
    }

    template <typename Fn>
    void for_each(Fn fn) const
    {
        if (!overflow.empty()) {
            for (const PinBox& box : overflow) {
                fn(box);
            }
            return;
        }
        for (int i = 0; i < inline_count; ++i) {
            fn(inline_boxes[i]);
        }
    }
};

struct PinGridVotes {
    std::array<PinGrid, kInlinePinBoxCapacity> inline_grids{};
    std::array<int, kInlinePinBoxCapacity> inline_counts{};
    int inline_count = 0;
    std::vector<PinGrid> overflow_grids;
    std::vector<int> overflow_counts;

    void add(const PinGrid& grid)
    {
        if (overflow_grids.empty()) {
            for (int i = 0; i < inline_count; ++i) {
                if (inline_grids[i] == grid) {
                    ++inline_counts[i];
                    return;
                }
            }
            if (inline_count < kInlinePinBoxCapacity) {
                inline_grids[inline_count] = grid;
                inline_counts[inline_count] = 1;
                ++inline_count;
                return;
            }
            overflow_grids.reserve(kInlinePinBoxCapacity * 2);
            overflow_counts.reserve(kInlinePinBoxCapacity * 2);
            for (int i = 0; i < inline_count; ++i) {
                overflow_grids.emplace_back(inline_grids[i]);
                overflow_counts.emplace_back(inline_counts[i]);
            }
            inline_count = 0;
        }

        for (int i = 0; i < static_cast<int>(overflow_grids.size()); ++i) {
            if (overflow_grids[i] == grid) {
                ++overflow_counts[i];
                return;
            }
        }
        overflow_grids.emplace_back(grid);
        overflow_counts.emplace_back(1);
    }

    PinGrid best_grid() const
    {
        PinGrid best{0, 0};
        int best_votes = -1;
        auto consider = [&](const PinGrid& grid, int votes) {
            if (votes > best_votes) {
                best_votes = votes;
                best = grid;
            }
        };
        if (!overflow_grids.empty()) {
            for (int i = 0; i < static_cast<int>(overflow_grids.size()); ++i) {
                consider(overflow_grids[i], overflow_counts[i]);
            }
        } else {
            for (int i = 0; i < inline_count; ++i) {
                consider(inline_grids[i], inline_counts[i]);
            }
        }
        return best;
    }
};

}  // namespace

OpenroadPinMapStats resolve_openroad_timer_pins(const GTDatabase& gtdb,
                                                int num_pins,
                                                std::vector<db::Pin*>& pin_id_to_dbpin,
                                                int threads)
{
    OpenroadPinMapStats stats;
    const int worker_count = std::max(1, threads);
    std::vector<int> direct_counts(worker_count, 0);

#pragma omp parallel for num_threads(worker_count) schedule(static)
    for (int net_idx = 0; net_idx < static_cast<int>(gtdb.rawdb.nets.size()); ++net_idx) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        db::Net* dbnet = gtdb.rawdb.nets[net_idx];
        if (dbnet == nullptr) {
            continue;
        }
        int local_direct = 0;
        for (db::Pin* dbpin : dbnet->pins) {
            if (dbpin == nullptr) {
                continue;
            }
            if (dbpin->gpdb_id >= 0 &&
                dbpin->gpdb_id < num_pins &&
                pin_id_to_dbpin[dbpin->gpdb_id] == nullptr) {
                pin_id_to_dbpin[dbpin->gpdb_id] = dbpin;
                ++local_direct;
            }
        }
        direct_counts[tid] += local_direct;
    }
    for (int count : direct_counts) {
        stats.gpdb_direct_pins += count;
    }

    int unresolved_pins = 0;
#pragma omp parallel for num_threads(worker_count) schedule(static) reduction(+:unresolved_pins)
    for (int pin_id = 0; pin_id < static_cast<int>(pin_id_to_dbpin.size()); ++pin_id) {
        if (pin_id_to_dbpin[pin_id] == nullptr) {
            ++unresolved_pins;
        }
    }
    stats.unresolved_pins = unresolved_pins;

    if (stats.unresolved_pins <= 0) {
        return stats;
    }

    std::unordered_map<std::string, db::Pin*> pin_name_to_dbpin;
    pin_name_to_dbpin.reserve(static_cast<std::size_t>(stats.unresolved_pins) * 4);
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
            stats.name_resolved_pins++;
        }
    }
    return stats;
}

bool openroad_pin_route_loc(const GTDatabase& gtdb,
                            const std::vector<db::Pin*>& pin_id_to_dbpin,
                            const std::vector<const db::Layer*>& routing_level_to_layer,
                            const OpenroadInferredGrid& openroad_grid,
                            int pin_id,
                            OpenroadPinRouteLoc& loc)
{
    loc = OpenroadPinRouteLoc{};
    if (pin_id < 0 || pin_id >= static_cast<int>(pin_id_to_dbpin.size()) ||
        pin_id_to_dbpin[pin_id] == nullptr) {
        return false;
    }
    db::Pin* pin = pin_id_to_dbpin[pin_id];
    if (pin->type == nullptr) {
        return false;
    }

    PinBoxList conn_boxes;
    int conn_layer = 0;
    auto add_box = [&](int routing_layer, int lx, int ly, int hx, int hy) {
        loc.pin_x = lx;
        loc.pin_y = ly;
        loc.pin_layer = routing_layer;
        if (routing_layer > conn_layer) {
            conn_layer = routing_layer;
            conn_boxes.clear();
        }
        if (routing_layer == conn_layer) {
            conn_boxes.push(PinBox{lx, ly, hx, hy});
        }
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

    if (conn_boxes.empty()) {
        return false;
    }

    loc.conn_layer = conn_layer;

    const db::Layer* conn_db_layer = nullptr;
    if (conn_layer >= 0 && conn_layer < static_cast<int>(routing_level_to_layer.size())) {
        conn_db_layer = routing_level_to_layer[conn_layer];
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
            conn_boxes.for_each([&](const PinBox& box) {
                min_coord = std::min(min_coord, horizontal ? box[1] : box[0]);
                max_coord = std::max(max_coord, horizontal ? box[3] : box[2]);
            });
            if (min_coord <= max_coord &&
                static_cast<float>(max_coord - min_coord) / static_cast<float>(track.step) <= 3.0f) {
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
                const bool first_inside = nearest_track >= min_coord && nearest_track <= max_coord;
                const bool second_inside = nearest_track2 >= min_coord && nearest_track2 <= max_coord;
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

    auto box_center = [&](const PinBox& box) {
        int cx = box[0] + (box[2] - box[0]) / 2;
        int cy = box[1] + (box[3] - box[1]) / 2;
        if (adjust_single_track) {
            if (adjust_horizontal) {
                cy = adjusted_track;
            } else {
                cx = adjusted_track;
            }
        }
        return PinGrid{cx, cy};
    };

    if (conn_boxes.size() == 1) {
        const PinGrid center = box_center(conn_boxes.front());
        const PinGrid grid = openroad_position_on_inferred_grid(gtdb.rawdb,
                                                                openroad_grid,
                                                                center.first,
                                                                center.second);
        loc.grid_x = grid.first;
        loc.grid_y = grid.second;
        loc.grid_src_x = center.first;
        loc.grid_src_y = center.second;
        loc.valid = true;
        return true;
    }

    PinGridVotes grid_votes;
    conn_boxes.for_each([&](const PinBox& box) {
        const PinGrid center = box_center(box);
        const PinGrid grid = openroad_position_on_inferred_grid(gtdb.rawdb,
                                                                openroad_grid,
                                                                center.first,
                                                                center.second);
        grid_votes.add(grid);
    });

    const PinGrid best_grid = grid_votes.best_grid();
    loc.grid_x = best_grid.first;
    loc.grid_y = best_grid.second;

    bool found_source = false;
    conn_boxes.for_each([&](const PinBox& box) {
        if (found_source) {
            return;
        }
        const PinGrid center = box_center(box);
        const PinGrid grid = openroad_position_on_inferred_grid(gtdb.rawdb,
                                                                openroad_grid,
                                                                center.first,
                                                                center.second);
        if (grid.first == loc.grid_x && grid.second == loc.grid_y) {
            loc.grid_src_x = center.first;
            loc.grid_src_y = center.second;
            found_source = true;
        }
    });

    loc.valid = true;
    return true;
}

}  // namespace openroad_rc
}  // namespace gt
