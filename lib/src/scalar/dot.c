#include <scalar.h>

// float dot_scalar(const float* x, const float* y, size_t n) {
//     int i     = 0;
//     float dot = 0.0f;
//     while(i < n) {
//         dot += y[i] * x[i];
//         i++;
//     }
//     return dot;
// }

float dot_scalar(const float *x, const float *y, size_t n)
{
    int i = 0;
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    while (i + 3 < n)
    {
        sum0 += x[i] * y[i];
        sum1 += x[i + 1] * y[i + 1];
        sum2 += x[i + 2] * y[i + 2];
        sum3 += x[i + 3] * y[i + 3];
        i += 4;
    }
    float dot = sum0 + sum1 + sum2 + sum3;
    while (i < n)
    {
        dot += x[i] * y[i];
        i++;
    }
    return dot;
}