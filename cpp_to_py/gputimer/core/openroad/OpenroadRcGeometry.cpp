#include "gputimer/core/openroad/OpenroadRcInternal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gt {
namespace openroad_rc {

bool route_point_matches(const OpenroadRoutePt& pt, const OpenroadRoutePtKey& key)
{
    return pt.valid && pt.x == key.x && pt.y == key.y && pt.layer == key.layer;
}

int openroad_layer_track_spacing(const db::Layer& layer)
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

int openroad_gcell_tile_size(const db::Database& rawdb)
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

std::pair<int, int> openroad_position_on_grid(const db::Database& rawdb,
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

int positive_mod(int value, int modulus)
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

int infer_openroad_grid_step(std::vector<int> coords)
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

int infer_openroad_grid_origin(const std::vector<int>& coords,
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

int infer_openroad_grid_origin_from_first(int first_coord,
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

void add_openroad_route_grid_point(OpenroadRouteGridStats& stats, int x, int y)
{
    if (!stats.have_x) {
        stats.have_x = true;
        stats.first_x = x;
    } else {
        stats.x_step = std::gcd(stats.x_step, std::abs(x - stats.first_x));
    }
    if (!stats.have_y) {
        stats.have_y = true;
        stats.first_y = y;
    } else {
        stats.y_step = std::gcd(stats.y_step, std::abs(y - stats.first_y));
    }
}

OpenroadInferredGrid infer_openroad_route_grid_from_stats(
    const std::vector<OpenroadRouteGridStats>& thread_stats,
    const db::Database& rawdb,
    int fallback_tile_size)
{
    bool have_x = false;
    bool have_y = false;
    int first_x = 0;
    int first_y = 0;
    int x_step = 0;
    int y_step = 0;
    for (const OpenroadRouteGridStats& stats : thread_stats) {
        if (stats.have_x) {
            if (!have_x) {
                have_x = true;
                first_x = stats.first_x;
                x_step = stats.x_step;
            } else {
                x_step = std::gcd(x_step, std::abs(stats.first_x - first_x));
                x_step = std::gcd(x_step, stats.x_step);
            }
        }
        if (stats.have_y) {
            if (!have_y) {
                have_y = true;
                first_y = stats.first_y;
                y_step = stats.y_step;
            } else {
                y_step = std::gcd(y_step, std::abs(stats.first_y - first_y));
                y_step = std::gcd(y_step, stats.y_step);
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

OpenroadInferredGrid infer_openroad_route_grid(
    const std::vector<std::unique_ptr<LocalSpefNetRc>>& local_nets,
    const db::Database& rawdb,
    int fallback_tile_size,
    int threads)
{
    const int worker_count = std::max(1, threads);
    std::vector<OpenroadRouteGridStats> thread_stats(worker_count);

#pragma omp parallel for num_threads(worker_count) schedule(static)
    for (int tid = 0; tid < worker_count; ++tid) {
        OpenroadRouteGridStats& stats = thread_stats[tid];
        const std::size_t net_begin =
            (local_nets.size() * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(worker_count);
        const std::size_t net_end =
            (local_nets.size() * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(worker_count);
        for (std::size_t net_idx = net_begin; net_idx < net_end; ++net_idx) {
            const auto& local_ptr = local_nets[net_idx];
            if (!local_ptr) {
                continue;
            }
            for (const OpenroadRoutePt& pt : local_ptr->route_points) {
                if (pt.valid) {
                    add_openroad_route_grid_point(stats, pt.x, pt.y);
                }
            }
        }
    }

    return infer_openroad_route_grid_from_stats(thread_stats, rawdb, fallback_tile_size);
}

std::pair<int, int> openroad_position_on_inferred_grid(
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

NangateLayerRc nangate45_layer_rc(int routing_level)
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

float nangate45_via_res_ohm(int lower_routing_level)
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

std::tuple<int, int, int, int> orient_box_for_iopin(int orient,
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

std::tuple<int, int, int, int> orient_box_for_cell(const db::CellType* ctype,
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

}  // namespace openroad_rc
}  // namespace gt
