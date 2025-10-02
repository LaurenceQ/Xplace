#include "Lut.h"

namespace gt {

bool is_time_lut_var(LutVar v) {
    switch (v) {
        case LutVar::INPUT_NET_TRANSITION:
        case LutVar::CONSTRAINED_PIN_TRANSITION:
        case LutVar::RELATED_PIN_TRANSITION:
        case LutVar::INPUT_TRANSITION_TIME:
            return true;
            break;

        default:
            return false;
            break;
    }
}

bool is_capacitance_lut_var(LutVar v) {
    switch (v) {
        case LutVar::TOTAL_OUTPUT_NET_CAPACITANCE:
            return true;
            break;

        default:
            return false;
            break;
    }
}

bool Lut::is_scalar() const {
    return indices1.size() == 1 && indices2.size() == 1;
}

inline bool Lut::empty() const {
    return indices1.size() == 0 && indices2.size() == 0;
}

float Lut::operator()(float val1, float val2) const {
  
    if(indices1.size() < 1 || indices2.size() < 1) {
        logger.info("invalid lut indices size");
    }

    // Interpolation
    constexpr auto interpolate = [] (float x, float x1, float x2, float y1, float y2) {

        assert(x1 < x2);

        if(x >= std::numeric_limits<float>::max() || x <= std::numeric_limits<float>::lowest()) {
        return x;
        }

        float slope = (y2 - y1) / (x2 - x1);

        if(x < x1) return y1 - (x1 - x) * slope;                  // Extrapolation.
        else if(x > x2)  return y2 + (x - x2) * slope;            // Extrapolation.
        else if(x == x1) return y1;                               // Boundary case.
        else if(x == x2) return y2;                               // Boundary case.
        else return y1 + (x - x1) * slope;                        // Interpolation.
    };

    // Case 1: scalar
    if(is_scalar()) return table[0];

    int idx1[2], idx2[2];

    idx1[1] = std::lower_bound(indices1.begin(), indices1.end(), val1) - indices1.begin();
    idx2[1] = std::lower_bound(indices2.begin(), indices2.end(), val2) - indices2.begin();


    // Case 2: linear inter/extra polation.
    idx1[1] = std::max(1, std::min(idx1[1], (int)(indices1.size() - 1)));
    idx2[1] = std::max(1, std::min(idx2[1], (int)(indices2.size() - 1)));
    idx1[0] = idx1[1] - 1;
    idx2[0] = idx2[1] - 1;

    //printf("Perform the linear interpolation on val1=%.5f (%d %d) and val2=%.5f (%d %d)\n", 
    //        val1, idx1[0], idx1[1], val2, idx2[0], idx2[1]);

    // 1xN array (N>=2)
    if(indices1.size() == 1) {  
        return interpolate(
        val2, 
        indices2[idx2[0]], 
        indices2[idx2[1]], 
        table[idx2[0]],
        table[idx2[1]]
        );
    }
    // Nx1 array (N>=2)
    else if(indices2.size() == 1) {   
        return interpolate(
        val1, 
        indices1[idx1[0]], 
        indices1[idx1[1]], 
        table[idx1[0]*indices2.size()], 
        table[idx1[1]*indices2.size()]
        );
    }
    // NxN array (N>=2)
    else {      
        float numeric[2];
        
        numeric[0] = interpolate(
        val1, 
        indices1[idx1[0]], 
        indices1[idx1[1]], 
        table[idx1[0]*indices2.size() + idx2[0]],
        table[idx1[1]*indices2.size() + idx2[0]]
        );

        numeric[1] = interpolate(
        val1, 
        indices1[idx1[0]], 
        indices1[idx1[1]], 
        table[idx1[0]*indices2.size() + idx2[1]],
        table[idx1[1]*indices2.size() + idx2[1]]
        );

        return interpolate(val2, indices2[idx2[0]], indices2[idx2[1]], numeric[0], numeric[1]);
    }
}
};  // namespace gt
