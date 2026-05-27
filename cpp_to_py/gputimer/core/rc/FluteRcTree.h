#include "gputimer/core/GPUTimer.h"
#include "common/utils/utils.h"
#include "common/db/Database.h"
#include "gputimer/db/GTDatabase.h"
#include <flute.hpp>
using namespace flt;

namespace gt {
class TimingTorchRawDB;
auto& retrieve_pins_from_pos(std::map<utils::PointT<int>, std::set<int>>& pos2pins_map, const utils::PointT<int>& point, int& index);
tuple<vector<int>, vector<int>, vector<float>, vector<int>, vector<int>, vector<int>, int, int> FluteRCTree(TimingTorchRawDB& timing_raw_db,
                                                                                                            float rf,
                                                                                                            float cf);
}