#include "DmpCeff.h"
#include "GPUTimer.h"
#include "gputiming.h"
#include "gputimer/db/GTDatabase.h"
// #include "utils.cuh"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
namespace gt {
enum DmpParam { t0,
                dt,
                ceff };    
enum DmpFunc { y20,
               y50,
               ipi };
__host__ dmp_model::dmp_model(GPUTimer* timer)
        : flat_net2pin_start_map(timer -> flat_net2pin_start_map), 
        flat_net2pin_map(timer -> flat_net2pin_map), 
        pin2net_map(timer -> pin2net_map),
        num_pins(timer -> num_pins), 
        num_nets(timer -> num_nets),
        level_list(timer -> level_list),
        pin_forward_arc_list_end(timer -> pin_forward_arc_list_end),
        pin_forward_arc_list(timer -> pin_forward_arc_list),
        timing_arc_to_pin_id(timer -> timing_arc_to_pin_id),
        pin_backward_arc_list_end(timer -> pin_backward_arc_list_end),
        pin_backward_arc_list(timer -> pin_backward_arc_list),
        timing_arc_from_pin_id(timer -> timing_arc_from_pin_id),
        arc_types(timer -> arc_types),
        arc_id2test_id(timer -> arc_id2test_id),
        // C1(timer -> h_dmp_rc_ -> C1),
        // C2(timer -> h_dmp_rc_ -> C2),
        // r_pi(timer -> h_dmp_rc_ -> r_pi),
        // elmore_delay(timer -> h_dmp_rc_ -> elmore_delay),
        pinSlew(timer -> pinSlew),
        pinAt(timer -> pinAT),
        pinRat(timer -> pinRAT),
        ceff(timer -> pinLoad),
        testRelatedAT(timer -> testRelatedAT),
        testRAT(timer -> testRAT),
        testConstraint(timer -> testConstraint),
        arcDelay(timer -> arcDelay),
        timing_arc_id_map(timer -> timing_arc_id_map),
        at_prefix_pin(timer -> at_prefix_pin),
        at_prefix_arc(timer -> at_prefix_arc),
        at_prefix_attr(timer -> at_prefix_attr),
        clock_period(timer -> clock_period),
        d_allocator(timer -> d_allocator),
        debug_on(timer -> dmp_debug_on)
        {
    cudaMalloc(&k0_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&k1_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&k2_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&k3_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&k4_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&p1_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&p2_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&p3_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&A_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&B_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&D_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&rd_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&t0, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&dt, sizeof(float) * num_pins * NUM_ATTR);
    // cudaMalloc(&ceff, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&vo_delay_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&vo_slew_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&z1_, sizeof(float) * num_pins * NUM_ATTR);
    cudaMalloc(&pin_ids, sizeof(int) * num_pins * NUM_ATTR * 2);
    cudaMalloc(&arc_ids, sizeof(int) * num_pins * NUM_ATTR * 2);
    cudaMemset(rd_, -1, sizeof(float) * num_pins * NUM_ATTR);
    // host-side
    const char** host_pin_ptrs = new const char*[num_pins];
    for (int i = 0; i < num_pins; i++) {
        size_t len = (timer->gtdb).pin_names[i].length() + 1;
        const char* dev_str;
        cudaMalloc(&dev_str, len);
        cudaMemcpy((void *)dev_str, (timer->gtdb).pin_names[i].c_str(), len, cudaMemcpyHostToDevice);
        host_pin_ptrs[i] = dev_str; // 保存 device 地址
    }
    cudaMalloc(&pin_names, sizeof(const char*) * num_pins);
    cudaMemcpy(pin_names, host_pin_ptrs, sizeof(const char*) * num_pins, cudaMemcpyHostToDevice);
    delete[] host_pin_ptrs;
    // host-side
    const char** host_net_ptrs = new const char*[num_nets];
    for (int i = 0; i < num_nets; i++) {
        size_t len = (timer->gtdb).net_names[i].length() + 1;
        const char* dev_str;
        cudaMalloc(&dev_str, len);
        cudaMemcpy((void *)dev_str, (timer->gtdb).net_names[i].c_str(), len, cudaMemcpyHostToDevice);
        host_net_ptrs[i] = dev_str; // 保存 device 地址
    }
    cudaMalloc(&net_names, sizeof(const char*) * num_nets);
    cudaMemcpy(net_names, host_net_ptrs, sizeof(const char*) * num_nets, cudaMemcpyHostToDevice);
    delete[] host_net_ptrs;
}
__host__ dmp_model::~dmp_model(){
    if(k0_){
        // cudaFree(C1);
        // cudaFree(C2);
        // cudaFree(r_pi);
        cudaFree(k0_);
        cudaFree(k1_);
        cudaFree(k2_); 
        cudaFree(k3_);
        cudaFree(k4_);
        cudaFree(p1_);
        cudaFree(p2_);
        cudaFree(p3_);
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(D_);
        cudaFree(rd_);
        cudaFree(t0);
        cudaFree(dt);
        cudaFree(ceff);
        // C1 = C2 = r_pi = nullptr;
    } 
}
void GPUTimer::initialize_dmp_model(){
    // cudaMemcpy(h_dmp_rc_, dmp_rc_, sizeof(dmp_rc), cudaMemcpyDeviceToHost);
    h_dmp_db = new dmp_model(this);
    cudaMalloc(&dmp_db, sizeof(dmp_model));
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(dmp_model), cudaMemcpyHostToDevice);
    
}
// __device__ void dmp_model::compute_pi_model(int net_id, int el_rf){
//     int start_id = flat_net2pin_start_map[net_id];
//     int end_id = flat_net2pin_start_map[net_id + 1];
//     int root = flat_net2pin_map[start_id];
//     double y1 = pinLoad[root * NUM_ATTR + el_rf];
//     double y2 = 0;
//     double y3 = 0;
//     for(int i = start_id + 1; i < end_id; i++){
//         int pin_id = flat_net2pin_map[i];
//         double cap = pinLoad[pin_id * NUM_ATTR + el_rf];
//         double res = pinRootRes[pin_id * NUM_ATTR + el_rf];
//         double y2_ = cap * cap * res;
//         y2 += -y2_;
//         y3 += y2_ * res * cap;
//     }
//     if(y3 <= 1e-10){
//         r_pi[net_id * NUM_ATTR + el_rf] = 0;
//         C1[net_id * NUM_ATTR + el_rf] = 0;
//         C2[net_id * NUM_ATTR + el_rf] = 0;
//     }
//     else{
//         C1[net_id * NUM_ATTR + el_rf] = static_cast<float>(y2 * y2 / y3); // 远端电容
//         C2[net_id * NUM_ATTR + el_rf] = static_cast<float>(y1 - y2 * y2 / y3); // 近端电容
//         if (C2[net_id * NUM_ATTR + el_rf] < 0.0)
//           C2[net_id * NUM_ATTR + el_rf] = 0.0;
//         r_pi[net_id * NUM_ATTR + el_rf] = static_cast<float>(-y3 * y3 / (y2 * y2 * y2));
//     }
//     // printf("net_id:%d el_rf:%d root:%d rpi:%.4f C1:%.4f C2:%.4f rootLoad:%.4f\n", net_id, el_rf, root, r_pi[net_id*NUM_ATTR+el_rf], C1[net_id*NUM_ATTR+el_rf], C2[net_id*NUM_ATTR+el_rf], pinLoad[root * NUM_ATTR+el_rf]);

// }

// __global__ void compute_pi_model_kernel(dmp_model *dmp_db){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int net_id = idx >> 2;
//     int el_rf = idx & (NUM_ATTR - 1);
//     if(net_id < dmp_db -> num_nets){
//         dmp_db -> compute_pi_model(net_id, el_rf);
//     }
// }
// void compute_pi_model_cuda(dmp_model *dmp_db, int num_nets){
//     compute_pi_model_kernel<<<BLOCK_NUMBER(num_nets * NUM_ATTR), BLOCK_SIZE>>>(dmp_db);
// }


__device__ double
dmp_model::voCrossingUpperBound(int pin_idx){
  return t0[pin_idx] + dt[pin_idx] + (C1[pin_idx] + C2[pin_idx]) * (rd_[pin_idx] + r_pi[pin_idx]) * 2.0;
}

__device__ double exp2(double x){
    if (x < -12.0)
        // exp(-12) = 6.1e-6
        return 0.0;
    else {
        double y = 1.0 + x / 4096.0;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        y *= y;
        return y;
    }    
}
__device__ double
dmp_model::y0(double t, double rd, 
           double cl)
{
  return t - rd * cl * (1.0 - exp2(-t / (rd * cl)));
}

__device__ double
dmp_model::y(double t,
          double t0,
          double dt,
          double rd, 
          double cl)
{
  double t1 = t - t0;
  if (t1 <= 0.0)
    return 0.0;
  else if (t1 <= dt)
    return y0(t1, rd, cl) / dt;
  else
    return (y0(t1, rd, cl) - y0(t1 - dt, rd, cl)) / dt;
}

__device__ double
dmp_model::y0dt(double t,
            double rd,
            double cl)
{
  return 1.0 - exp2(-t / (rd * cl));
}

__device__ double
dmp_model::y0dcl(double t,
                double rd,
                double cl)
{
  return rd * ((1.0 + t / (rd * cl)) * exp2(-t / (rd * cl)) - 1);
}
__device__ void
dmp_model::dy(double t,
           double t0,
           double dt,
           double rd,
           double cl,
           // Return values.
           double &dydt0,
           double &dyddt,
           double &dydcl)
{
  double t1 = t - t0;
  if (t1 <= 0.0)
    dydt0 = dyddt = dydcl = 0.0;
  else if (t1 <= dt) {
    dydt0 = -y0dt(t1, rd, cl) / dt;
    dyddt = -y0(t1, rd, cl) / (dt * dt);
    dydcl = y0dcl(t1, rd, cl) / dt;
  }
  else {
    dydt0 = -(y0dt(t1, rd, cl) - y0dt(t1 - dt, rd, cl)) / dt;
    dyddt = -(y0(t1, rd, cl) + y0(t1 - dt, rd, cl)) / (dt * dt) + y0dt(t1 - dt, rd, cl) / dt;
    dydcl = (y0dcl(t1, rd, cl) - y0dcl(t1 - dt, rd, cl)) / dt;
  }
}





__device__ void dmp_model::Vl0(int pin_idx, double t, double &vl, double &dvl_dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx % NUM_ATTR;
    int arc_id = arc_ids[idx];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    double p3_ = 1.0 / elmore_delay[to_pin_id * NUM_ATTR + i];
    double D1 = k0_[pin_idx] * (k1_[pin_idx] - k2_[pin_idx] / p3_);
    double D3 = -p3_ * k0_[pin_idx] * k3_[pin_idx] / (p1_[pin_idx] - p3_);
    double D4 = -p3_ * k0_[pin_idx] * k4_[pin_idx] / (p2_[pin_idx] - p3_);
    double D5 = k0_[pin_idx] * (k2_[pin_idx] / p3_ - k1_[pin_idx] + p3_ * k3_[pin_idx] / (p1_[pin_idx] - p3_) + p3_ * k4_[pin_idx] / (p2_[pin_idx] - p3_));
    double exp_p1 = exp2(-p1_[pin_idx] * t);
    double exp_p2 = exp2(-p2_[pin_idx] * t);
    double exp_p3 = exp2(-p3_ * t);
    vl = D1 + t + D3 * exp_p1 + D4 * exp_p2 + D5 * exp_p3;
    dvl_dt = 1.0 - D3 * p1_[pin_idx] * exp_p1 - D4 * p2_[pin_idx] * exp_p2 - D5 * p3_ * exp_p3;
}
__device__ void
dmp_model::V0(
          int pin_idx,
          double t,
          // Return values.
          double &vo,
          double &dvo_dt)
{
    double exp_p1 = exp2(-p1_[pin_idx] * t);
    double exp_p2 = exp2(-p2_[pin_idx] * t);
    vo = k0_[pin_idx] * (k1_[pin_idx] + k2_[pin_idx] * t + k3_[pin_idx] * exp_p1 + k4_[pin_idx] * exp_p2);
    dvo_dt = k0_[pin_idx] * (k2_[pin_idx] - k3_[pin_idx] * p1_[pin_idx] * exp_p1 - k4_[pin_idx] * p2_[pin_idx] * exp_p2);
}

__device__ void dmp_model::Vl(double t, double &vl, double &dvl_dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pin_idx = pin_ids[idx];
    double t1 = t - t0[pin_idx];
    if(t1 <= 0.0){
        vl = 0.0;
        dvl_dt = 0.0;
    }
    else if(t1 <= dt[pin_idx]){
        double vl0, dvl0_dt;
        Vl0(pin_idx, t1, vl0, dvl0_dt);
        vl = vl0 / dt[pin_idx];
        dvl_dt = dvl0_dt / dt[pin_idx];
    }
    else{
        double vl0, dvl0_dt;
        Vl0(pin_idx, t1, vl0, dvl0_dt);
        double vl0_dt, dvl0_dt_dt;
        Vl0(pin_idx, t1 - dt[pin_idx], vl0_dt, dvl0_dt_dt);
        vl = (vl0 - vl0_dt) / dt[pin_idx];
        dvl_dt = (dvl0_dt - dvl0_dt_dt) / dt[pin_idx];
    }
}
__device__ void
dmp_model::Vo(double t,
           // Return values.
           double &vo,
           double &dvo_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pin_idx = pin_ids[idx];
    double t1 = t - t0[pin_idx];
    if (t1 <= 0.0) {
        vo = 0.0;
        dvo_dt = 0.0;
    }
    else if (t1 <= dt[pin_idx]) {
        double v0, dv0_dt;
        V0(pin_idx, t1, v0, dv0_dt);

        vo = v0 / dt[pin_idx];
        dvo_dt = dv0_dt / dt[pin_idx];
    }
    else {
        double v0, dv0_dt;
        V0(pin_idx, t1, v0, dv0_dt);

        double v0_dt, dv0_dt_dt;
        V0(pin_idx, t1 - dt[pin_idx], v0_dt, dv0_dt_dt);

        vo = (v0 - v0_dt) / dt[pin_idx];
        dvo_dt = (dv0_dt - dv0_dt_dt) / dt[pin_idx];
    }
}
__device__ void dmp_model::vl_func(double vth, double t, double &y, double &dy){
    double vl, vl_dt;
    Vl(t, vl, vl_dt);
    y = vl - vth; // goal: y = 0, y = vl(t) - vth
    dy = vl_dt;
}
__device__ void dmp_model::vo_func(double vth, double t, double &y, double &dy){
    double vo, vo_dt;
    Vo(t, vo, vo_dt);
    y = vo - vth;
    dy = vo_dt;
}
__device__ double dmp_model::findRoot_vo(double vth, double x1, double x2){
    double y1, y2, dy;
    vo_func(vth, x1, y1, dy);
    vo_func(vth, x2, y2, dy);
    if(y1 * y2 > 0.0) return nanf(""); // cannot find root
    if(y1 == 0.0) return x1;
    if(y2 == 0.0) return x2;
    if(y1 > 0.0){
        double xtemp = x1; x1 = x2; x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    vo_func(vth, root, y, dy);
    for(int iter = 0; iter < MAX_ITER; iter++){
        // Newton/raphson out of range.
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0)
        // Not decreasing fast enough.
        || (abs(2.0 * y) > abs(dx_prev * dy))) { // step too large
        // Bisect x1/x2 interval.
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        }
        else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= x_tol * abs(root)) {
            // Converged.
            return root;
        }

        vo_func(vth, root, y, dy);
        if (y < 0.0)
            x1 = root;
        else
            x2 = root;
    }
    return nanf("");
}
__device__ double dmp_model::findRoot_vl(double vth, double x1, double x2){ // TODO: solve non-deterministic
    double y1, y2, dy;
    vl_func(vth, x1, y1, dy);
    vl_func(vth, x2, y2, dy);
    if(y1 * y2 > 0.0) return nanf(""); // cannot find root
    if(y1 == 0.0) return x1;
    if(y2 == 0.0) return x2;
    if(y1 > 0.0){
        double xtemp = x1; x1 = x2; x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    vl_func(vth, root, y, dy);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int arc_id = arc_ids[idx];
    int to_pin_id = timing_arc_to_pin_id[arc_id];

    // if(to_pin_id == 4706){
    //     int pin_idx = pin_ids[idx];
        // printf("Rd = %.4f t0 = %.4f dt = %.4f\n        "
        //     "k0 = %.4f k1 = %.4f k3 = %.4f k4 = %.4f p1 = %.4f p2 = %.4f p3 = %.4f\n        ",
        //     rd_[pin_idx], t0[pin_idx], dt[pin_idx], 
        //     k0_[pin_idx], k1_[pin_idx], k3_[pin_idx], k4_[pin_idx], p1_[pin_idx], p2_[pin_idx], p3_[pin_idx]         
        // );
    // }
    for(int iter = 0; iter < MAX_ITER; iter++){
        // if(to_pin_id == 4706)printf("iter:%d root:%.4f y:%.4f dy:%.4f x1:%.4f x2:%.4f dx:%.4f dx_prev:%.4f\n", iter, root, y, dy, x1, x2, dx, dx_prev);

        // Newton/raphson out of range.
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0)
        // Not decreasing fast enough.
        || (abs(2.0 * y) > abs(dx_prev * dy))) { // step too large
        // Bisect x1/x2 interval.
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        }
        else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= x_tol * abs(root)) {
            // Converged.
            return root;
        }

        vl_func(vth, root, y, dy);
        if (y < 0.0)
            x1 = root;
        else
            x2 = root;
    }
    return nanf("");
}
// __device__ void DmpAlg::showVl()
// {
//   report_->reportLine("  t    vl(t)");
//   double ub = vlCrossingUpperBound();
//   for (double t = t0_; t < t0_ + ub * 2.0; t += ub / 10.0) {
//     double vl, dvl_dt;
//     Vl(t, vl, dvl_dt);
//     report_->reportLine(" %g %g", t, vl);
//   }
// }

__device__ double dmp_model::findVlCrossing(double vth, double t_lower, double t_upper){
    double t_vth = findRoot_vl(vth, t_lower, t_upper);
    if(isnan(t_vth)){
        int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int from_pin_id = timing_arc_from_pin_id[arc_id];
        int to_pin_id = timing_arc_to_pin_id[arc_id];
        double y1, y2, dy; 
        vl_func(vth, t_lower, y1, dy);
        vl_func(vth, t_upper, y2, dy);
        if(debug_on)printf("Error: cannot find Vl crossing point from:%s to:%s vth:%.3f t_vth:%e x1:%e x2:%e y1:%.4f y2:%.4f rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_names[from_pin_id], pin_names[to_pin_id], vth, t_vth, t_lower, t_upper, y1, y2, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        return nanf("");
    }
    else return t_vth;
}
__device__ double 
dmp_model::findVoCrossing(double vth,
                       double t_lower,
                       double t_upper){
    double t_vth = findRoot_vo(vth, t_lower, t_upper);
    if(isnan(t_vth)){
        int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int from_pin_id = timing_arc_from_pin_id[arc_id];
        int to_pin_id = timing_arc_to_pin_id[arc_id];
        if(debug_on)printf("Error: cannot find Vo crossing point from:%s to:%s rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_names[from_pin_id], pin_names[to_pin_id], rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        return nanf("");
    }
    return t_vth;
}
__device__ void dmp_model::propagateLoadSlewDelay(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx & 0b11;
    int arc_id = arc_ids[idx];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    bool error_flag = false;
    double elmore = elmore_delay[to_pin_id * NUM_ATTR + i];
    pin_ids[idx] = from_pin_id * NUM_ATTR + i; 
    int pin_idx = pin_ids[idx];
    float si = pinSlew[pin_idx];
    if (isnan(si)) return;
    // float imp = pinImpulse[to_pin_id * NUM_ATTR + i];
    // float so = si < 0.0 ? -sqrt(si * si + imp * imp) : sqrt(si * si + imp * imp); // todo: how to calc slew? 
    pinSlew[to_pin_id * NUM_ATTR + i] = si;
    int el_rf_rf = (i << 1) + (i & 1);  // same rise/fall for two pins in net connections
    arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = elmore;

    if(!isnan(rd_[pin_idx]) && rd_[pin_idx] > 0.0 && !isnan(vo_delay_[pin_idx])){
        p3_[pin_idx] = 1.0 / elmore; // non deterministic!!!
        double t_lower = t0[pin_idx];
        double t_upper = voCrossingUpperBound(pin_idx) + elmore * 2.0;
        double load_delay = findVlCrossing(vth_, t_lower, t_upper); // time point voltage reach middle
        double tl = findVlCrossing(vl_, t_lower, load_delay); // time point voltage reach low
        double th = findVlCrossing(vh_, load_delay, t_upper); // time point voltage reach high
        double delay1 = load_delay - vo_delay_[pin_idx];
        double slew1 = (th - tl) / slew_derate_;

        // if((el_rf_rf & 1) == 1 && (el_rf_rf >> 2) == 1 && !isnan(slew1) && !isnan(delay1) && !isnan(load_delay) && delay1 > 0.01 && to_pin_id == 4706){ 
           
        //     const char* el_str = (el_rf_rf >> 2) ? "MAX" : "MIN";
        //     const char frf_c = (el_rf_rf & 1) ? 'v' : '^';
        //     const char trf_c = (el_rf_rf & 1) ? 'v' : '^';
        //     printf("%s %s %c -> %s %c to_pin_id = %d\n        "
        //         "C2 = %.4f r_pi = %.4f C1 = %.4f elmore = %.4f\n        ",
        //         el_str, pin_names[from_pin_id], frf_c, pin_names[to_pin_id], trf_c, to_pin_id,
        //         C2[pin_idx], r_pi[pin_idx], C1[pin_idx], elmore
        //         );
        //     printf("Rd = %.4f t0 = %.4f dt = %.4f\n        "
        //            "k0 = %.4f k1 = %.4f k3 = %.4f k4 = %.4f p1 = %.4f p2 = %.4f p3 = %.4f\n        ",
        //             rd_[pin_idx], t0[pin_idx], dt[pin_idx], 
        //             k0_[pin_idx], k1_[pin_idx], k3_[pin_idx], k4_[pin_idx], p1_[pin_idx], p2_[pin_idx], p3_[pin_idx]         
        //         );
        //     printf(
        //         "t_upper = %.4f t_lower = %.4f load_delay = %.4f tl = %.4f th = %.4f\n      "
        //         "si = %.4f so = %.4f delay = %.4f\n", 
        //         t_upper, t_lower, load_delay, tl, th,
        //         si, slew1, delay1
        //     );
        // }
        if(isnan(slew1) || isnan(delay1)){
            error_flag = true;
            if(debug_on)printf("Error: load slew or delay is nan net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        }        
        else{
            if(delay1 < 0.0){
                if(-delay1 > vth_time_tol * vo_delay_[pin_idx]){
                    error_flag = true;
                    if(debug_on)printf("Error: load delay less than 0 net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
                }
                else delay1 = elmore;
            }
            if(slew1 < si){
                if((si - slew1) > vth_time_tol * si){
                    error_flag = true;
                    if(debug_on)printf("Error: load slew less than driver slew net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
                }
                else slew1 = pinSlew[from_pin_id * NUM_ATTR + i];
            }
        }
        if(!error_flag) {
            pinSlew[to_pin_id * NUM_ATTR + i] = slew1;
            arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = delay1;
        }
    }
    else if(pin_backward_arc_list_end[from_pin_id] == pin_backward_arc_list_end[from_pin_id + 1]){ // primary input
        
        arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = -elmore * log(1.0 - vth_);
        pinSlew[to_pin_id * NUM_ATTR + i] = pinSlew[from_pin_id * NUM_ATTR + i] + elmore * log((1.0 - vl_) / (1.0 - vh_)) / slew_derate_;
        // printf("from:%s to:%s elmore:%.4f delay:%.4f slew:%.4f C1:%.4f C2:%.4f r_pi:%.4f\n", pin_names[from_pin_id], pin_names[to_pin_id], elmore, arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf], pinSlew[to_pin_id * NUM_ATTR + i], C1[pin_idx], C2[pin_idx], r_pi[pin_idx]);
    }

}



__device__ void 
dmp_model::gateCapDelaySlew(double lc, double &delay, double &slew){ // maybe there is some better way to do this
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;    
    int el = i >> 2;                            // early late
    int fel_rf = i >> 1;                        // from early/late rise/fall
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);  // to early/late rise/fall
    int irf = fel_rf & 1;                       // input rise/fall
    int orf = tel_rf & 1;                       // output rise/fall
    int arc_id = arc_ids[idx];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    // int to_pin_id = timing_arc_to_pin_id[arc_id];

    if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
    float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
    int timing_id = timing_arc_id_map[arc_id * 2 + el];
    slew = d_allocator->query(timing_id, irf, orf, si, lc, 1);  // slew output = LUT(slew input, load capacitance)
    delay = d_allocator->query(timing_id, irf, orf, si, lc, 0);

}

__device__ void 
dmp_model::gateDelays(double ceff, double &t_vth, double &t_vl, double &slew){
    gateCapDelaySlew(ceff, t_vth, slew);
    t_vl = t_vth - slew * (vth_ - vl_) / (vh_ - vl_);
}




__device__ void dmp_model::gateModelRd(int pin_idx, double d1, double s1){
    double cap1 = C1[pin_idx] + C2[pin_idx];
    double cap2 = cap1 + 1;
    double d2, s2;
    gateCapDelaySlew(cap2, d2, s2);
    rd_[pin_idx] = -log(vth_) * fabs(d1 - d2) / (cap2 - cap1); // dak : strange, should be 1/log(vth), seems like rd is not independent of inslew
    // double cap = C1[pin_idx] + C2[pin_idx];
    // rd_[pin_idx] = s1 / cap / log(vh_ / vl_);
    // if(rd_[pin_idx] > 0.1)
        // printf("pin_idx:%d rd:%.4f cap1:%.4f cap2:%.4f d1:%.4f d2:%.4f s1:%.4f s2:%.4f\n", pin_idx, rd_[pin_idx], C1[pin_idx], C2[pin_idx], d1, d2, s1, s2);
    // printf("pin_idx:%d rd:%.4f cap:%.4f slew:%.4f\n", pin_idx, rd_[pin_idx], cap, s1);
}
__device__ void dmp_model::init_dmp_factors(int pin_idx){
    z1_[pin_idx] = 1.0 / (r_pi[pin_idx] * C1[pin_idx]);
    k0_[pin_idx] = 1.0 / (rd_[pin_idx] * C2[pin_idx]);
    double a = r_pi[pin_idx] * rd_[pin_idx] * C1[pin_idx] * C2[pin_idx];
    double b = rd_[pin_idx] * (C1[pin_idx] + C2[pin_idx]) + r_pi[pin_idx] * C1[pin_idx];
    double sqrt_ = sqrt(b * b - 4 * a);
    p1_[pin_idx] = (b + sqrt_) / (2 * a);
    p2_[pin_idx] = (b - sqrt_) / (2 * a);

    double p1p2 = (p1_[pin_idx] * p2_[pin_idx]);
    k2_[pin_idx] = z1_[pin_idx] / p1p2;
    k1_[pin_idx] = (1.0 - k2_[pin_idx] * (p1_[pin_idx] + p2_[pin_idx])) / p1p2;
    k4_[pin_idx] = (k1_[pin_idx] * p1_[pin_idx] + k2_[pin_idx]) / (p2_[pin_idx] - p1_[pin_idx]);
    k3_[pin_idx] = -k1_[pin_idx] - k4_[pin_idx];

    double z_ = (C1[pin_idx] + C2[pin_idx]) / (r_pi[pin_idx] * C1[pin_idx] * C2[pin_idx]);
    A_[pin_idx] = z_ / p1p2;
    B_[pin_idx] = (z_ - p1_[pin_idx]) / (p1_[pin_idx] * (p1_[pin_idx] - p2_[pin_idx]));
    D_[pin_idx] = (z_ - p2_[pin_idx]) / (p2_[pin_idx] * (p2_[pin_idx] - p1_[pin_idx]));
}
// Eqn 13, Eqn 14.
__device__ double
dmp_model::ipiIceff(int pin_idx, double dt, double ceff_time, double ceff)
{
  double exp_p1_dt = exp2(-p1_[pin_idx] * ceff_time);
  double exp_p2_dt = exp2(-p2_[pin_idx] * ceff_time);
  double exp_dt_rd_ceff = exp2(-ceff_time / (rd_[pin_idx] * ceff));
  double ipi = (A_[pin_idx] * ceff_time + (B_[pin_idx] / p1_[pin_idx]) * (1.0 - exp_p1_dt) + (D_[pin_idx] / p2_[pin_idx]) * (1.0 - exp_p2_dt)) / (rd_[pin_idx] * ceff_time * dt);
  double iceff = (rd_[pin_idx] * ceff * ceff_time - (rd_[pin_idx] * ceff) * (rd_[pin_idx] * ceff) * (1.0 - exp_dt_rd_ceff)) / (rd_[pin_idx] * ceff_time * dt);
  return ipi - iceff;
}
__device__ bool 
dmp_model::evalDmpEqns(double *x_, double *fvec_, double (*fjac_)[3]){
    double t0 = x_[DmpParam::t0];
    double dt = x_[DmpParam::dt];
    double ceff = x_[DmpParam::ceff];
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    if (ceff < 0.0){
        if(debug_on)printf("Error: eqn eval failed: ceff < 0\n");
        return false;
    }
    if (ceff > (C1[pin_idx] + C2[pin_idx])){
        if(debug_on)printf("Error: eqn eval failed: ceff > c2 + c1\n");
        return false;
    }

    double t_vth, t_vl, slew;
    gateDelays(ceff, t_vth, t_vl, slew);
    if (slew == 0.0){
        if(debug_on)printf("Error: eqn eval failed: slew = 0\n");
        return false;
    }

    double ceff_time = slew / (vh_ - vl_);
    if (ceff_time > 1.4 * dt)
        ceff_time = 1.4 * dt;

    if (dt <= 0.0){
        if(debug_on)printf("Error: eqn eval failed: dt < 0\n");
        return false;
    }
    double exp_p1_dt = exp2(-p1_[pin_idx] * dt);
    double exp_p2_dt = exp2(-p2_[pin_idx] * dt);
    double exp_dt_rd_ceff = exp2(-dt / (rd_[pin_idx] * ceff));

    double y50 = y(t_vth, t0, dt, rd_[pin_idx], ceff);
    // Match Vl.
    double y20 = y(t_vl, t0, dt, rd_[pin_idx], ceff);
    fvec_[DmpFunc::ipi] = ipiIceff(pin_idx, dt, ceff_time, ceff);
    fvec_[DmpFunc::y50] = y50 - vth_;
    fvec_[DmpFunc::y20] = y20 - vl_;
    fjac_[DmpFunc::ipi][DmpParam::t0] = 0.0;
    fjac_[DmpFunc::ipi][DmpParam::dt] =
        (-A_[pin_idx] * dt + B_[pin_idx] * dt * exp_p1_dt - (2 * B_[pin_idx] / p1_[pin_idx]) * (1.0 - exp_p1_dt) + D_[pin_idx] * dt * exp_p2_dt - (2 * D_[pin_idx] / p2_[pin_idx]) * (1.0 - exp_p2_dt) + rd_[pin_idx] * ceff * (dt + dt * exp_dt_rd_ceff - 2 * rd_[pin_idx] * ceff * (1.0 - exp_dt_rd_ceff))) / (rd_[pin_idx] * dt * dt * dt);
    fjac_[DmpFunc::ipi][DmpParam::ceff] =
        (2 * rd_[pin_idx] * ceff - dt - (2 * rd_[pin_idx] * ceff + dt) * exp2(-dt / (rd_[pin_idx] * ceff))) / (dt * dt);

    dy(t_vl, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y20][DmpParam::t0], fjac_[DmpFunc::y20][DmpParam::dt], fjac_[DmpFunc::y20][DmpParam::ceff]);

    dy(t_vth, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y50][DmpParam::t0], fjac_[DmpFunc::y50][DmpParam::dt], fjac_[DmpFunc::y50][DmpParam::ceff]);

    return true;
}
// luDecomp, luSolve based on MatClass from C. R. Birchenhall,
// University of Manchester
// ftp://ftp.mcc.ac.uk/pub/matclass/libmat.tar.Z

// Crout's Method of LU decomposition of square matrix, with implicit
// partial pivoting.  A is overwritten. U is explicit in the upper
// triangle and L is in multiplier form in the subdiagionals i.e. subdiag
// a[i,j] is the multiplier used to eliminate the [i,j] term.
//
// Replaces a[0..size-1][0..size-1] by the LU decomposition.
// index[0..size-1] is an output vector of the row permutations.
// Return error msg on failure.
__device__ bool
luDecomp(double (*a)[3], const int size, int *index,
         // Temporary supplied by caller.
         // scale stores the implicit scaling of each row.
         double *scale){
  // Find implicit scaling factors.
    for (int i = 0; i < size; i++) {
        double big = 0.0;
        for (int j = 0; j < size; j++) {
        double temp = abs(a[i][j]);
        if (temp > big)
            big = temp;
        }
        if (big == 0.0){
            printf("Error: LU decomposition no non-zero row element\n");
            return false;
        }
        scale[i] = 1.0 / big;
    }
    int size_1 = size - 1;
    for (int j = 0; j < size; j++) {
        // Run down jth column from top to diag, to form the elements of U.
        for (int i = 0; i < j; i++) {
            double sum = a[i][j];
            for (int k = 0; k < i; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
        }
        // Run down jth subdiag to form the residuals after the elimination
        // of the first j-1 subdiags.  These residuals diviyded by the
        // appropriate diagonal term will become the multipliers in the
        // elimination of the jth. subdiag. Find index of largest scaled
        // term in imax.
        double big = 0.0;
        int imax = 0;
        for (int i = j; i < size; i++) {
            double sum = a[i][j];
            for (int k = 0; k < j; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            double dum = scale[i] * abs(sum);
            if (dum >= big) {
                big = dum;
                imax = i;
            }
        }
        // Permute current row with imax.
        if (j != imax) {
        // Yes, do so...
            for (int k = 0; k < size; k++) {
                double dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            scale[imax] = scale[j];
        }
        index[j] = imax;
        // If diag term is not zero divide subdiag to form multipliers.
        if (a[j][j] == 0.0)
            a[j][j] = 1e-12;
        if (j != size_1) {
            double pivot = 1.0 / a[j][j];
            for (int i = j + 1; i < size; i++)
                a[i][j] *= pivot;
        }
    }
    return true;
}

// Solves the set of size linear equations a*x=b, assuming A is LU form
// but assume b has not been transformed.
//  a[0..size-1] is LU decomposition
// Returns the solution vector x in b.
// a and index are not modified.
__device__ void luSolve(double (*a)[3], const int size, const int *index, double b[]){
// Transform b allowing for leading zeros.
    int non_zero = -1;
    for (int i = 0; i < size; i++) {
        int iperm = index[i];
        double sum = b[iperm];
        b[iperm] = b[i];
        if (non_zero != -1) {
            for (int j = non_zero; j <= i - 1; j++)
                sum -= a[i][j] * b[j];
        }
        else {
            if (sum != 0.0)
                non_zero = i;
        }
        b[i] = sum;
    }
    // Backsubstitution.
    for (int i = size - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < size; j++)
            sum -= a[i][j] * b[j];
        b[i] = sum / a[i][i];
    }
}
__device__ bool dmp_model::newtonRaphson(int max_iter, int size, double *x, double (*fjac)[3], double *fvec, int *index, double *p, double *scale){
    for (int k = 0; k < max_iter; k++) {
        bool error = !evalDmpEqns(x, fvec, fjac);
        if(debug_on){
            int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
            int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
            int from_pin_id = timing_arc_from_pin_id[arc_id];
            int to_pin_id = timing_arc_to_pin_id[arc_id];        
            double t0 = x[DmpParam::t0];
            double dt = x[DmpParam::dt];
            double ceff = x[DmpParam::ceff];
            printf("k=%d from:%s to:%s ceff:%e dt:%e t0:%e fvec:ipi:%e y50:%e y20:%e\n", k, pin_names[from_pin_id], pin_names[to_pin_id], ceff, dt, t0, fvec[DmpFunc::ipi], fvec[DmpFunc::y50], fvec[DmpFunc::y20]);
        }
        if(error)return false;
        for (int i = 0; i < size; i++)
        // Right-hand side of linear equations.
            p[i] = -fvec[i];
        error |= !luDecomp(fjac, size, index, scale);
        if(error)return false;
        luSolve(fjac, size, index, p);

        bool all_under_x_tol = true;
        for (int i = 0; i < size; i++) {
            if (abs(p[i]) > abs(x[i]) * x_tol)
                all_under_x_tol = false;
            x[i] += p[i];
        }
        if (all_under_x_tol){
            // printf("NR converged in %d iterations\n", k);
            return true;
        }
    }
    if(debug_on)printf("Error: Newton-Raphson exceeded maximum iterations\n");
    return false;
}

__device__ void dmp_model::findDriverParams(double delay, double slew, bool &error_flag){
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    ceff[pin_idx] = C1[pin_idx] + C2[pin_idx];
    // double t_vl = delay - slew * (vth_ - vl_) / (vh_ - vl_);
    dt[pin_idx] = slew / (vh_ - vl_);
    t0[pin_idx] = delay + log(1.0 - vth_) * rd_[pin_idx] * ceff[pin_idx] - vth_ * dt[pin_idx];
    double x_[3]; // hard code pi_model dimension is 3
    double fjac_[3][3];
    double fvec_[3];
    int index[3];
    double p[3];
    double scale[3] = {1.0, 1.0, 1.0};
    x_[DmpParam::t0] = t0[pin_idx];
    x_[DmpParam::dt] = dt[pin_idx];
    x_[DmpParam::ceff] = ceff[pin_idx];
    error_flag |= !newtonRaphson(100, 3, x_, fjac_, fvec_, index, p, scale);
    t0[pin_idx] = x_[DmpParam::t0];
    dt[pin_idx] = x_[DmpParam::dt];
    ceff[pin_idx] = x_[DmpParam::ceff];
    
}

__device__ void
dmp_model::findDriverDelaySlew(int pin_idx, double &delay, double &slew){
  double t_upper = voCrossingUpperBound(pin_idx);
  delay = findVoCrossing(vth_, t0[pin_idx], t_upper);
  if(isnan(delay)){
    delay = slew = nanf("");
    return ;
  }
  double tl = findVoCrossing(vl_, t0[pin_idx], delay);
  double th = findVoCrossing(vh_, delay, t_upper);
  if(isnan(tl) || isnan(th)){
    delay = slew = nanf("");
    return ;
  }
  // Convert measured slew to table slew.
  slew = (th - tl) / slew_derate_;
}
__device__ void dmp_model::propagateGateSlewDelay(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;    
    int el = i >> 2;                            // early late
    int fel_rf = i >> 1;                        // from early/late rise/fall
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);  // to early/late rise/fall
    // int irf = fel_rf & 1;                       // input rise/fall
    // int orf = tel_rf & 1;                       // output rise/fall
    int arc_id = arc_ids[idx];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    pin_ids[idx] = from_pin_id * NUM_ATTR + tel_rf; // TODO: race condition!! all input pins are setting one net!
    int pin_idx = pin_ids[idx];
    C1[pin_idx] = C1[to_pin_id * NUM_ATTR + tel_rf];
    C2[pin_idx] = C2[to_pin_id * NUM_ATTR + tel_rf];
    r_pi[pin_idx] = r_pi[to_pin_id * NUM_ATTR + tel_rf];
    bool error_flag = false;
    double so, delay;
    so = nanf("");
    delay = nanf("");
    gateCapDelaySlew(C1[pin_idx] + C2[pin_idx], delay, so);
    if(isnan(so) || isnan(delay))return ;
    gateModelRd(pin_idx, delay, so);
    if(rd_[pin_idx] <= r_pi[pin_idx] * 1000){
        init_dmp_factors(pin_idx);
        findDriverParams(delay, so, error_flag);
        // const char* el_str = el ? "MAX" : "MIN";
        // const char frf_c = (fel_rf & 1) ? 'v' : '^';
        // const char trf_c = (tel_rf & 1) ? 'v' : '^';
        // printf("%s %s %c -> %s %c\n        DMP si = %.4f C2 = %.4f r_pi = %.4f C1 = %.4f Rd = %.4f \n        t0 = %.4f dt = %.4f ceff = %.4f \n        gate delay = %.4f so = %.4f \n", 
        //         el_str, pin_names[from_pin_id], frf_c, pin_names[to_pin_id], trf_c, pinSlew[from_pin_id * NUM_ATTR + fel_rf], C2[pin_idx],  r_pi[pin_idx], C1[pin_idx], rd_[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx], delay, so);
        // if(el == 0 && fel_rf == 0 && tel_rf == 0)
        //     printf("%d -> %d DMP ceff:%.4f\n", from_pin_id, to_pin_id, ceff[pin_idx]);

        if(!error_flag){
            gateCapDelaySlew(ceff[pin_idx], delay, so);
            double vo_delay, vo_slew;
            findDriverDelaySlew(pin_idx, vo_delay, vo_slew);
            if(isnan(vo_delay) || isnan(vo_slew)) {
                error_flag = true;
            }
            else {
                vo_delay_[pin_idx] = vo_delay;
                so = vo_slew_[pin_idx] = vo_slew;
            }
        }
        // pinSlew[to_pin_id * NUM_ATTR + tel_rf] = vo_slew;
    }
    else error_flag = true;

    arcDelay[arc_id * 2 * NUM_ATTR + i] = delay;
    if (isnan(pinSlew[to_pin_id * NUM_ATTR + tel_rf]) || ((pinSlew[to_pin_id * NUM_ATTR + tel_rf] > so) ^ el)) {  // setup: max output slew. hold: min output slew.
        atomicExch(&pinSlew[to_pin_id * NUM_ATTR + tel_rf], so);
        atomicExch(&k0_[to_pin_id * NUM_ATTR + tel_rf], k0_[pin_idx]);
        atomicExch(&k1_[to_pin_id * NUM_ATTR + tel_rf], k1_[pin_idx]);
        atomicExch(&k2_[to_pin_id * NUM_ATTR + tel_rf], k2_[pin_idx]);
        atomicExch(&k3_[to_pin_id * NUM_ATTR + tel_rf], k3_[pin_idx]);
        atomicExch(&k4_[to_pin_id * NUM_ATTR + tel_rf], k4_[pin_idx]);
        atomicExch(&p1_[to_pin_id * NUM_ATTR + tel_rf], p1_[pin_idx]);
        atomicExch(&p2_[to_pin_id * NUM_ATTR + tel_rf], p2_[pin_idx]);
        // atomicExch(&p3_[to_pin_id * NUM_ATTR + tel_rf], p3_[pin_idx]);
        atomicExch(&z1_[to_pin_id * NUM_ATTR + tel_rf], z1_[pin_idx]);
        atomicExch(&A_[to_pin_id * NUM_ATTR + tel_rf], A_[pin_idx]);
        atomicExch(&B_[to_pin_id * NUM_ATTR + tel_rf], B_[pin_idx]);
        atomicExch(&D_[to_pin_id * NUM_ATTR + tel_rf], D_[pin_idx]);
        atomicExch(&rd_[to_pin_id * NUM_ATTR + tel_rf], rd_[pin_idx]);
        atomicExch(&t0[to_pin_id * NUM_ATTR + tel_rf], t0[pin_idx]);
        atomicExch(&dt[to_pin_id * NUM_ATTR + tel_rf], dt[pin_idx]);
        atomicExch(&ceff[to_pin_id * NUM_ATTR + tel_rf], ceff[pin_idx]);
        atomicExch(&vo_delay_[to_pin_id * NUM_ATTR + tel_rf], vo_delay_[pin_idx]);        
    }
    if(error_flag){
        vo_delay_[pin_idx] = nanf("");
    } 
    // else printf("pin_idx:%d, arc_id:%d, from_pin:%d to_pin:%d, c1:%.3f c2:%.3f ceff:%.3f, rd:%.3f, rpi:%.3f t0:%.3f, dt:%.3f, delay:%.3f, slew:%.3f\n", pin_idx, arc_id, from_pin_id, to_pin_id, C1[pin_idx], C2[pin_idx], ceff[pin_idx], rd_[pin_idx], r_pi[pin_idx], t0[pin_idx], dt[pin_idx], delay, so);
    
}

__device__ void dmp_model::propagateSlewDelay(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int arc_id = arc_ids[idx];
    int arc_type = arc_types[arc_id];
    if ((arc_type == 0) && (i < NUM_ATTR)) {  // 0 for net arc
        propagateLoadSlewDelay();
        // int from_pin_id = timing_arc_from_pin_id[arc_id];
        // int to_pin_id = timing_arc_to_pin_id[arc_id];
        // float si = pinSlew[from_pin_id * NUM_ATTR + (i >> 1)];
        // float so = pinSlew[to_pin_id * NUM_ATTR + (i & (0b11))];
        // float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        // if((arc_type == 0) && (i < NUM_ATTR))delay = arcDelay[arc_id * 2 * NUM_ATTR + ((i << 1) + (i & 1))]; // net arc delay use same rise/fall
        // printf("arc_id:%d from:%s to:%s #in_arc:%d arc_type:%d i:%d si:%.4f so:%.4f delay:%.4f\n", arc_id, pin_names[from_pin_id], pin_names[to_pin_id], pin_backward_arc_list_end[from_pin_id+1] - pin_backward_arc_list_end[from_pin_id], arc_type, i, si, so, delay);

    } else if (arc_type == 1) {                     // 1 for gate arc
        propagateGateSlewDelay();
    }
    else return ;
}
__device__ void dmp_model::propagateAT(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int arc_id = arc_ids[idx];
    int arc_type = arc_types[arc_id];
    int i = idx & 0b111;
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    int el = i >> 2;
    int fel_rf = i >> 1;
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);
    int irf = fel_rf & 1;
    int orf = tel_rf & 1;
    if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
    float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
    float at = pinAt[from_pin_id * NUM_ATTR + fel_rf] + delay;

    // FIXME: conflict
    if (isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || ((pinAt[to_pin_id * NUM_ATTR + tel_rf] > at) ^ el)) {
        atomicExch(&pinAt[to_pin_id * NUM_ATTR + tel_rf], at);
        at_prefix_pin[to_pin_id * NUM_ATTR + tel_rf] = from_pin_id;
        at_prefix_arc[to_pin_id * NUM_ATTR + tel_rf] = arc_id;
        at_prefix_attr[to_pin_id * NUM_ATTR + tel_rf] = fel_rf;
    }
}
__device__ void dmp_model::propagateTest(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int arc_id = arc_ids[idx];
    int test_id = arc_id2test_id[arc_id];
    if(test_id == -1)return ;
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    if (i < NUM_ATTR) {
        const int el = i >> 1;
        const int rf = i & 1;
        const int el_rf_rf = (i << 1) + (i & 1);
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || (isnan(pinSlew[to_pin_id * NUM_ATTR + i]))) return;
        int fel = el ^ 1;  // clock -> data. clock late -> data early (hold)
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        int frf = d_allocator->d_is_rising_edge_triggered[timing_id] ? 0 : 1;
        if (frf && !d_allocator->d_is_falling_edge_triggered[timing_id]) {
            return;
        }
        const int fel_rf = (fel << 1) + frf;
        if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;

        if (el == 0) {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf];
        } else {
            testRelatedAT[test_id * NUM_ATTR + i] = pinAt[from_pin_id * NUM_ATTR + fel_rf] + (frf ? 0.5 * clock_period : clock_period);  // setup is checked at next cycle (first cycle for triggering 1st FF)
        }

        float sr = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float sc = pinSlew[to_pin_id * NUM_ATTR + i];   
        testConstraint[test_id * NUM_ATTR + i] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);
        if (!isnan(testConstraint[test_id * NUM_ATTR + i]) && !isnan(testRelatedAT[test_id * NUM_ATTR + i])) {
            if (el == 0) {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] + testConstraint[test_id * NUM_ATTR + i];  // hold clocks needs data stay late, rat = at_clk + T_hold
            } else {
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] - testConstraint[test_id * NUM_ATTR + i];  // setup clock needs data come early, rat = at_clk - T_setup
            }
            testRAT[test_id * NUM_ATTR + i] = pinRat[to_pin_id * NUM_ATTR + i];
        }
    }    
}
__device__ void dmp_model::propagatePin(int to_pin_idx){
    int to_pin = level_list[to_pin_idx];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (index_type i = pin_backward_arc_list_end[to_pin]; i < pin_backward_arc_list_end[to_pin + 1]; i++) {
        // index_type arc_id = pin_backward_arc_list[i];
        // index_type from_pin_id = timing_arc_from_pin_id[arc_id];
        // int arc_type = arc_types[arc_id];
        arc_ids[idx] = pin_backward_arc_list[i];
        propagateSlewDelay();
        
        propagateAT();
        if (clock_period > 0) {
            propagateTest();
        }
        // int test_id = arc_id2test_id[arc_id];
        // if (clock_period > 0 && test_id != -1) {
        //     propagateTest(test_id);
        // }
    }
}

__global__ void propagatePin_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = idx >> 3;
    if(pin_id < num_pins_level){
        dmp_db -> propagatePin(level_start_offset + pin_id);
    }
}
__device__ void dmp_model::updatePinRat(int arc_id, float *from_rats){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    for (int ti = threadIdx.x; ti < threadIdx.x + 2 * NUM_ATTR; ti++) {
        const int i = ti & 0b111;
        if (isnan(from_rats[ti])) continue;
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        float rat = from_rats[ti];
        if (!d_allocator->d_is_constraint[timing_id]) {
            if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || ((pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
                atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
            }
        } else {
            if (el == 0) {
                const int fel_rf = 2 + irf;
                const int tel_rf = orf;
                if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || (pinRat[from_pin_id * NUM_ATTR + fel_rf] > rat)) {
                    atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                }
            } else {
                const int fel_rf = irf;
                const int tel_rf = 2 + orf;
                if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || (pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat)) {
                    atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
                }
            }
        }
    }
}
__device__ void dmp_model::propagateRAT(int arc_id, float *from_rats){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int arc_type = arc_types[arc_id];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;                                                  // rat_f - delay, at_f - delay.
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) || ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {  // early(hold up): max rat
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        int el = i >> 2;
        int fel_rf = i >> 1;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int irf = fel_rf & 1;
        int orf = tel_rf & 1;
        if (timing_arc_id_map[arc_id * 2 + el] == -1) return;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (!d_allocator->d_is_constraint[timing_id]) {
            if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
            float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
            float rat = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
            from_rats[threadIdx.x] = rat;
        } else {  // CLK -> D
            if (!d_allocator->is_transition_defined(timing_id, irf, orf)) return;
            if (el == 0) {
                const int fel_rf = 2 + irf;
                const int tel_rf = orf;
                float at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
                if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || isnan(at)) return;
                float slack = (pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf]) * -1;
                float rat = at + slack;  // fel = 1 (setup)
                from_rats[threadIdx.x] = rat;
            } else {
                const int fel_rf = irf;
                const int tel_rf = 2 + orf;
                float at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
                if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) || isnan(at)) return;
                float slack = (pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf]);
                float rat = at - slack;
                from_rats[threadIdx.x] = rat;
            }
        }
    }    
}
__device__ void dmp_model::propagatePinBack(int level_idx, float *from_rats){
        index_type from_pin_id = level_list[level_idx];
        for (index_type i = pin_forward_arc_list_end[from_pin_id]; i < pin_forward_arc_list_end[from_pin_id + 1]; i++) {
            index_type arc_id = pin_forward_arc_list[i];
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int i = threadIdx.x; i < threadIdx.x + 2 * NUM_ATTR; i++) from_rats[i] = nanf("");
            }
            __syncthreads();

            propagateRAT(arc_id, from_rats);

            __syncthreads();
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                updatePinRat(arc_id, from_rats);

            }
        }
}
__global__ void propagatePinBack_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        dmp_db -> propagatePinBack(level_start_offset + pin_idx, from_rats);

    }
}
void update_timing_dmp_cuda(dmp_model* dmp_db, vector<int> level_list_end_cpu){

    for (int i = 1; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePin_dmp<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE>>>(dmp_db, level_start_offset, num_pins_level);
        gpuErrchk( cudaDeviceSynchronize() );
        // if(i == 2)break;
    }
    gpuErrchk( cudaDeviceSynchronize() );
    for (int i = level_list_end_cpu.size() - 3; i >= 0; i--) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        propagatePinBack_dmp<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(dmp_db, level_start_offset, num_pins_level);

        gpuErrchk( cudaDeviceSynchronize() );
    }
    gpuErrchk( cudaDeviceSynchronize() );   

}
void print_pinLoad_cuda(dmp_model* dmp_db, vector<int> level_list_end_cpu, vector<std::string> pin_names){
    int total_num_pins = level_list_end_cpu.back();
    assert(total_num_pins == (int)pin_names.size());
    float* pinLoad_host = new float[total_num_pins * NUM_ATTR];
    float* C1_host = new float[total_num_pins * NUM_ATTR];
    float* C2_host = new float[total_num_pins * NUM_ATTR];
    float* rd_host = new float[total_num_pins * NUM_ATTR];
    float* t0_host = new float[total_num_pins * NUM_ATTR];
    float* dt_host = new float[total_num_pins * NUM_ATTR];
    float* pinslew_host = new float[total_num_pins * NUM_ATTR];
    float* vodelay_host = new float[total_num_pins * NUM_ATTR];
    float* voslew_host = new float[total_num_pins * NUM_ATTR];
    float* pinAt_host = new float[total_num_pins * NUM_ATTR];
    float* pinRat_host = new float[total_num_pins * NUM_ATTR];
    int* level_pin_list_host = new int[total_num_pins];
    int* backward_arc_list_end_host = new int[total_num_pins + 1];
    dmp_model* dmp_db_host = new dmp_model();
    cudaMemcpy(dmp_db_host, dmp_db, sizeof(dmp_model), cudaMemcpyDeviceToHost);
    cudaMemcpy(backward_arc_list_end_host, dmp_db_host->pin_backward_arc_list_end, sizeof(int) * (total_num_pins + 1), cudaMemcpyDeviceToHost);
    int num_arcs = backward_arc_list_end_host[total_num_pins];
    printf("num_arcs = %d\n", num_arcs);
    int* backward_arc_list_host = new int[num_arcs];
    int* timing_arc_from_pin_id_host = new int[num_arcs];
    float* arcDelay_host = new float[num_arcs * 2 * NUM_ATTR];
    cudaMemcpy(timing_arc_from_pin_id_host, dmp_db_host->timing_arc_from_pin_id, sizeof(int) * num_arcs, cudaMemcpyDeviceToHost);
    cudaMemcpy(backward_arc_list_host, dmp_db_host->pin_backward_arc_list, sizeof(int) * num_arcs, cudaMemcpyDeviceToHost);
    cudaMemcpy(arcDelay_host, dmp_db_host->arcDelay, sizeof(float) * num_arcs * 2 * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(level_pin_list_host, dmp_db_host->level_list, sizeof(int) * total_num_pins, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinLoad_host, dmp_db_host->ceff, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_host, dmp_db_host->C1, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_host, dmp_db_host->C2, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(rd_host, dmp_db_host->rd_, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(t0_host, dmp_db_host->t0, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(dt_host, dmp_db_host->dt, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinslew_host, dmp_db_host->pinSlew, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(vodelay_host, dmp_db_host->vo_delay_, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(voslew_host, dmp_db_host->vo_slew_, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinAt_host, dmp_db_host->pinAt, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinRat_host, dmp_db_host->pinRat, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    for (int i = 0; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        printf("==== level %d ======= %d \n", i, num_pins_level);
        for(int j = 0; j < num_pins_level; j++){
            for(int attr = 0; attr < NUM_ATTR; attr++){
                int pin = level_pin_list_host[level_start_offset + j];
                if(!isnan(pinLoad_host[pin * NUM_ATTR + attr])){
                    printf("pin %s attr %d pin_at = %E pin_rat = %E load = %E C1 = %E C2 = %E rd = %E t0 = %E dt = %E pinslew = %E vodelay = %E voslew = %E\n", pin_names[pin].c_str(), attr, pinAt_host[pin * NUM_ATTR + attr], pinRat_host[pin * NUM_ATTR + attr], pinLoad_host[pin * NUM_ATTR + attr], C1_host[pin * NUM_ATTR + attr], C2_host[pin * NUM_ATTR + attr], rd_host[pin * NUM_ATTR + attr], t0_host[pin * NUM_ATTR + attr], dt_host[pin * NUM_ATTR + attr], pinslew_host[pin * NUM_ATTR + attr], vodelay_host[pin * NUM_ATTR + attr], voslew_host[pin * NUM_ATTR + attr]);
                }
                if(j > 0){
                    for(int k = backward_arc_list_end_host[pin] ; k < backward_arc_list_end_host[pin + 1]; k++){
                        int arc_id = backward_arc_list_host[k];
                        int from_pin = timing_arc_from_pin_id_host[arc_id];
                        assert(k < num_arcs);
                        assert(arc_id < num_arcs);
                        assert(from_pin < total_num_pins);
                        for(int frf = 0; frf < 2; frf++){
                            int arc_idx = arc_id * 2 * NUM_ATTR + ((attr & (1 << 1)) << 1) + (frf << 1) + (attr & 1);
                            assert(arc_idx < num_arcs * 2 * NUM_ATTR);
                            float delay = arcDelay_host[arc_idx];
                            printf("    from pin %s %c  arc_id = %d delay = %E\n", pin_names[from_pin].c_str(), frf == 0 ? '^' : 'v',   arc_id, delay);
                        }
                    }
                }
            }
        }
        gpuErrchk( cudaDeviceSynchronize() );
        // if(i == 2)break;
    }

}

}