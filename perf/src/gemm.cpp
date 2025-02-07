#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define GEMM_FUNC gemm_scalar
#elif defined(rvv)
#define MOD rvv
#define GEMM_FUNC gemm_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_gemm(int length, int iterations) {
    std::vector<float> src_1(length * length, 0);
    std::vector<float> src_2(length * length, 0);
    std::vector<float> dst(length * length, 0);

    // prepareVector(src_1.data(), length * length);
    // prepareVector(src_2.data(), length * length);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        GEMM_FUNC(src_1.data(), src_2.data(), dst.data(), length, length, length);
        end = rdcycle();
        diff = end - start;
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar GEMM with length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV GEMM with length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
}

int main() {

    //
    // Bench small sizes
    //
    // constexpr int small_length = 16;
    // constexpr int small_iterations = 5;
    // bench_gemm(small_length, small_iterations);

    // //
    // // Bench medium sizes
    // //
    // constexpr int medium_0_length = 48;
    // constexpr int medium_0_iterations = 1;
    // bench_gemm(medium_0_length, medium_0_iterations);

    // constexpr int medium_1_length = 128;
    // constexpr int medium_1_iterations = 1;
    // bench_gemm(medium_1_length, medium_1_iterations);

    // constexpr int medium_2_length = 256;
    // constexpr int medium_2_iterations = 1;
    // bench_gemm(medium_2_length, medium_2_iterations);

    // 
    // Bench big sizes
    //
    constexpr int a = 256;
    constexpr int b = 10;
    bench_gemm(a, b);

     constexpr int big_length = 501;
    constexpr int big_iterations = 10;
    bench_gemm(big_length, big_iterations);

    return 0;
}
