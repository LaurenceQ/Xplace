#include "DmpCeff.h"
#include "GPUTimer.h"

namespace gt {

void compute_pi_model_cuda(dmp_model* dmp_db, int num_nets);
void GPUTimer::compute_pi_model(){
    assert(dmp_db != nullptr);
    compute_pi_model_cuda(dmp_db, num_nets);
}

}