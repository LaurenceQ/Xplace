#pragma once

#include "DmpCeff.h"

namespace gt {

void dmp_debug_print_counts(dmp_model* dmp_db, const char* label);
void dmp_debug_print_first_level_sample(dmp_model* dmp_db, int level_idx, index_type level_start_offset);
void dmp_debug_print_parallel_stats(dmp_model* dmp_db,
                                    const vector<int>& level_list_end_cpu,
                                    const char* label);

} // namespace gt
