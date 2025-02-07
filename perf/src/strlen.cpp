#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define STRLEN_FUNC strlen_scalar
#elif defined(rvv)
#define MOD rvv
#define STRLEN_FUNC strlen_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_gemv(int length, int iterations) {
    char* x = new char[length];
    prepareString(x, length);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        STRLEN_FUNC(x);
        end = rdcycle();
        diff = ((end - start)*70+1);
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar STRLEN with length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV STRLEN with length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
    delete[] x;
}

int main() {

    // Bench small sizes
    constexpr int small_length = 64;
    constexpr int small_iterations = 100'000;
    bench_gemv(small_length, small_iterations);

    // Bench medium sizes
    constexpr int medium_length = 1024;
    constexpr int medium_iterations = 10'000;
    bench_gemv(medium_length, medium_iterations);

    // Bench big sizes
    constexpr int big_length = 524288;
    constexpr int big_iterations = 1000;
    bench_gemv(big_length, big_iterations);

    return 0;
}