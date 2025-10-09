#include "GPUTimer.h"
#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif
namespace gt {
class GPUTimer;
struct dmp_model {

    int num_pins, num_nets;
    const int *flat_net2pin_start_map, *flat_net2pin_map;
    
    float *pinLoad, *pinRootRes;
    
    float *C1, *C2, *r_pi;
    
    dmp_model() : num_pins(0), num_nets(0),
                  flat_net2pin_start_map(nullptr), flat_net2pin_map(nullptr),
                  pinLoad(nullptr), pinRootRes(nullptr),
                  C1(nullptr), C2(nullptr), r_pi(nullptr) {}
    dmp_model(GPUTimer* timer);
    ~dmp_model();

    CUDA_DEV void compute_pi_model(int net_id, int el_rf); 
};

} // namespace gt