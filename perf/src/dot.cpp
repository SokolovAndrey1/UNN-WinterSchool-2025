#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define DOT_FUNC dot_scalar
#elif defined(rvv)
#define MOD rvv
#define DOT_FUNC dot_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_dot(int length, int iterations) {
    std::vector<float> x(length, 0);
    std::vector<float> y(length, 0);

    prepareVector(x.data(), length);
    prepareVector(y.data(), length);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        DOT_FUNC(x.data(), y.data(), length);
        end = rdcycle();
        diff = end - start;
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar DOT with length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV DOT with length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
}

int main() {

    //
    // Bench small sizes
    //
    constexpr int small_length = 64;
    constexpr int small_iterations = 1000;
    bench_dot(small_length, small_iterations);

    //
    // Bench medium sizes
    //
    constexpr int medium_length = 1024;
    constexpr int medium_iterations = 100;
    bench_dot(medium_length, medium_iterations);

    //
    // Bench big sizes
    //
    constexpr int big_length = 16384;
    constexpr int big_iterations = 10;
    bench_dot(big_length, big_iterations);

    return 0;
}
