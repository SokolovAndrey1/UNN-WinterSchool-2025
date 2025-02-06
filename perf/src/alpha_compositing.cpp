#include "perf_common.hpp"

#if defined(scalar)
#define MOD scalar
#define ALPHA_COMPOSITING_FUNC alpha_compositing_scalar
#elif defined(rvv)
#define MOD rvv
#define ALPHA_COMPOSITING_FUNC alpha_compositing_rvv
#else
#error "Tested function is not defined"
#endif

static inline void bench_alpha_compositing(int width, int height, int channels, int iterations) {
    int length = width * height * channels;
    std::vector<uint8_t> foreground(length, 0);
    std::vector<uint8_t> background(length, 0);
    std::vector<uint8_t> dst(length, 0);
    float alpha = 0.5f;

    // prepareImage(foreground.data(), width, height, channels);
    // prepareImage(background.data(), width, height, channels);

    volatile uint64_t start = 0,
                    end = 0,
                    totalCycles = 0,
                    minCycles = std::numeric_limits<uint64_t>::max(),
                    diff;

    for(int i = 0; i < iterations; i++){
        start = rdcycle();
        ALPHA_COMPOSITING_FUNC(foreground.data(), background.data(), alpha, dst.data(), width, height, channels);
        end = rdcycle();
        diff = end - start;
        totalCycles += diff;
        if (diff < minCycles)
            minCycles = diff;
    }
#if defined(scalar)
    std::string funcName = std::string{"Scalar ALPHA_COMPOSITING with arr length "} + std::to_string(length);
#elif defined(rvv)
    std::string funcName = std::string{"RVV ALPHA_COMPOSITING with arr length "} + std::to_string(length);
#else
#error "Tested function is not defined"
#endif

    printStat(minCycles, totalCycles, iterations, length, funcName);
}

int main() {

    //
    // Bench small sizes
    //
    constexpr int small_width = 10;
    constexpr int small_iterations = 10'000;
    bench_alpha_compositing(small_width, small_width, small_width, small_iterations);

    //
    // Bench medium sizes
    //
    constexpr int medium_length = 50;
    constexpr int medium_iterations = 1'000;
    bench_alpha_compositing(medium_length, medium_length, medium_length, medium_iterations);

    //
    // Bench big sizes
    //
    constexpr int big_length = 100;
    constexpr int big_iterations = 100;
    bench_alpha_compositing(big_length, big_length, big_length, big_iterations);

    return 0;
}
