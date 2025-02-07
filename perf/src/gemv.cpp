#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define GEMV_FUNC gemv_scalar
#elif defined(rvv)
#define MOD rvv
#define GEMV_FUNC gemv_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_gemv(int length, int n, int iterations) {
    std::vector<float> A(length * n, 0);
    std::vector<float> x(length, 0);
    std::vector<float> y(length, 0);

    prepareMatrix(A.data(), length, n);
    prepareVector(x.data(), length);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        GEMV_FUNC(length, n, 1.5, A.data(), x.data(), y.data());
        end = rdcycle();
        diff = ((end - start)*70+1)/y.size();
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar GEMV with length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV GEMV with length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
}

int main() {

    // Bench small sizes
    constexpr int small_length = 64;
    constexpr int small_iterations = 1000;
    constexpr int small_n = 64;
    bench_gemv(small_length, small_n, small_iterations);
    
    // Bench medium sizes
    constexpr int medium_length = 1024;
    constexpr int medium_iterations = 100;
    constexpr int medium_n = 1024;
    bench_gemv(medium_length, medium_n, medium_iterations);
    
    // Bench big sizes
    constexpr int big_length = 8192;
    constexpr int big_iterations = 10;
    constexpr int big_n = 8192;
    bench_gemv(big_length, big_n, big_iterations);

    return 0;
}
