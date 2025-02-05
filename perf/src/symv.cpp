#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define SYMV_FUNC symv_scalar
#elif defined(rvv)
#define MOD rvv
#define SYMV_FUNC symv_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_symv(int length, int iterations) {
    std::vector<float> src(length, 0);
    std::vector<float> dst(length, 0);

    std::vector<float> a(length * length);
    static float alpha = 0;
    std::vector<float> x(length, 0);
    std::vector<float> y(length, 0);
    // prepareFloat(&alpha, 10);
    // prepareVector(a.data(), length * length);
    // prepareVector(x.data(), length);
    // prepareVector(y.data(), length);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        SYMV_FUNC(length, alpha, a.data(), x.data(), y.data());
        end = rdcycle();
        diff = end - start;
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar SYMV with length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV SYMV with length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
}

int main() {

    // // //
    // // // Bench small sizes
    // // //
    // // constexpr int small_length = 4;
    // // constexpr int small_iterations = 10;
    // // bench_symv(small_length, small_iterations);

    // // //
    // // // Bench medium sizes
    // // //
    // // constexpr int medium_length = 8;
    // // constexpr int medium_iterations = 20;
    // // bench_symv(medium_length, medium_iterations);
    
    // //
    // // Bench big sizes
    // //
    // constexpr int big_length = 1024;
    // constexpr int big_iterations = 1;
    // bench_symv(big_length, big_iterations);
    

    
    //
    // Bench small sizes
    //
    constexpr int small_length = 1024;
    constexpr int small_iterations = 1000;
    bench_symv(small_length, small_iterations);

    //
    // Bench medium sizes
    //
    constexpr int medium_length = 8192;
    constexpr int medium_iterations = 100;
    bench_symv(medium_length, medium_iterations);
    
    //
    // Bench big sizes
    //
    constexpr int big_length = 16384;
    constexpr int big_iterations = 50;
    bench_symv(big_length, big_iterations);
    

    return 0;
}
