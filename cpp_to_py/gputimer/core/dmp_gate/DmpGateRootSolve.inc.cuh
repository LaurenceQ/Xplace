__device__ __forceinline__ bool dmpSolve2x2(double a00,
                                            double a01,
                                            double a10,
                                            double a11,
                                            double b0,
                                            double b1,
                                            double& x0,
                                            double& x1) {
    const double det = a00 * a11 - a01 * a10;
    const double scale = fabs(a00 * a11) + fabs(a01 * a10) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return isfinite(x0) && isfinite(x1);
}

__device__ __forceinline__ bool dmpSolve3x3(double a00,
                                            double a01,
                                            double a02,
                                            double a10,
                                            double a11,
                                            double a12,
                                            double a20,
                                            double a21,
                                            double a22,
                                            double b0,
                                            double b1,
                                            double b2,
                                            double& x0,
                                            double& x1,
                                            double& x2) {
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a10 * a22 - a12 * a20;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 - a01 * c01 + a02 * c02;
    const double scale = fabs(a00 * c00) + fabs(a01 * c01) + fabs(a02 * c02) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    const double det0 = b0 * c00 - a01 * (b1 * a22 - a12 * b2) +
                        a02 * (b1 * a21 - a11 * b2);
    const double det1 = a00 * (b1 * a22 - a12 * b2) -
                        b0 * c01 +
                        a02 * (a10 * b2 - b1 * a20);
    const double det2 = a00 * (a11 * b2 - b1 * a21) -
                        a01 * (a10 * b2 - b1 * a20) +
                        b0 * c02;
    x0 = det0 / det;
    x1 = det1 / det;
    x2 = det2 / det;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
}

