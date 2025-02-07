#include "gemm_test.hpp"
#include <ctime>
#include <cstdlib>

constexpr size_t nIndex{0U};
constexpr size_t mIndex{1U};
constexpr size_t kIndex{2U};

template <typename T>
class GemmSquare : public ::testing::Test
{
public:
    static constexpr size_t n = std::tuple_element_t<nIndex, T>{};
    static constexpr size_t m = std::tuple_element_t<mIndex, T>{};
    static constexpr size_t k = std::tuple_element_t<kIndex, T>{};

    using ElemType = float;
    using VectorType = std::vector<ElemType>;
};

COMMON_TEST(SimpleTest_big) {
    const size_t n = 500;
    const size_t m = 500;
    const size_t k = 500;
    srand(time(NULL));
    float *A = (float*)calloc(n*m, sizeof(float));
    float *B = (float*)calloc(m*k, sizeof(float));
    for(int i = 0; i < m*n; i++)
    {
        A[i] = rand();
        B[i] = rand();
    }
    float C_ref[n*k] = {0.0f};
    float C_comp[n*k] = {0.0f};

    gemm_ref(A, B, C_ref, n, m, k);
    GEMM_FUNC(A, B, C_comp, n, m, k);

    ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, n, k, std::numeric_limits<float>::epsilon()));
}

// COMMON_TEST(SimpleTets_4x4) {
//     const size_t rows = 4;
//     const size_t cols = 4;
//     const float A[] = {1.0f,  2.0f,  3.0f,  4.0f,
//                        5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f,  10.0f, 11.0f, 12.0f,
//                        13.0f, 14.0f, 15.0f, 16.0f};
//     const float B[] = {1.0f,  2.0f,  3.0f,  4.0f,
//                        5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f,  10.0f, 11.0f, 12.0f,
//                        13.0f, 14.0f, 15.0f, 16.0f};
//     float C_ref[rows*cols] = {0.0f};
//     float C_comp[rows*cols] = {0.0f};

//     gemm_ref(A, B, C_ref, rows, cols, rows);

//     for(int i = 0; i < 4; i++)
//     {
//         for(int j = 0; j < 4; j++)
//             std::cout << C_ref[i*4 +j] << " ";
//     std::cout << "\n";
//     }
//     std::cout << "\n";
    
//     GEMM_FUNC(A, B, C_comp, rows, cols, rows);
//     for(int i = 0; i < 4; i++)
//     {
//         for(int j = 0; j < 4; j++)
//             std::cout << C_comp[i*4 +j] << " ";
//     std::cout << "\n";
//     }

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
// }

// COMMON_TEST(SimpleTest_4x5_x_5x1) {
//     const size_t n = 4;
//     const size_t m = 5;
//     const size_t k = 1;
//     const float A[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f/*, 21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f*/};
//     const float B[] = {1.0f,  2.0f,  3.0f,  4.0f,
//                        5.0f/*,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f,
//                        13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f,
//                        21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f,
//                        29.0f, 30.0f, 31.0f, 32.0f*/};
//     float C_ref[n*k] = {0.0f};
//     float C_comp[n*k] = {0.0f};

//     gemm_ref(A, B, C_ref, n, m, k);
//     GEMM_FUNC(A, B, C_comp, n, m, k);
//     for(int i = 0; i < n; i++)
//     {
//         for(int j = 0; j < k; j++)
//             std::cout << C_ref[i*k +j] << " ";
//     std::cout << "\n";
//     }
//     std::cout << "\n";

//      for(int i = 0; i < n; i++)
//     {
//         for(int j = 0; j < k; j++)
//             std::cout << C_comp[i*k +j] << " ";
//     std::cout << "\n";
//     }
//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, n, k, std::numeric_limits<float>::epsilon()));
// }

// COMMON_TEST(SimpleTest_4x8_x_8x4) {
//     const size_t n = 4;
//     const size_t m = 8;
//     const size_t k = 4;
//     const float A[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
//     const float B[] = {1.0f,  2.0f,  3.0f,  4.0f,
//                        5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f,
//                        13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f,
//                        21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f,
//                        29.0f, 30.0f, 31.0f, 32.0f};
//     float C_ref[n*k] = {0.0f};
//     float C_comp[n*k] = {0.0f};

//     gemm_ref(A, B, C_ref, n, m, k);
//     GEMM_FUNC(A, B, C_comp, n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, n, k, std::numeric_limits<float>::epsilon()));
// }

// COMMON_TEST(SimpleTest_8x4_x_4x8) {
//     const size_t n = 8;
//     const size_t m = 4;
//     const size_t k = 8;
//     const float A[] = {1.0f,  2.0f,  3.0f,  4.0f,
//                        5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f,
//                        13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f,
//                        21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f,
//                        29.0f, 30.0f, 31.0f, 32.0f};
//     const float B[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
//     float C_ref[n*k] = {0.0f};
//     float C_comp[n*k] = {0.0f};

//     gemm_ref(A, B, C_ref, n, m, k);
//     GEMM_FUNC(A, B, C_comp, n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, n, k, std::numeric_limits<float>::epsilon()));
// }


// TEST(GemmSquare, SimpleTets_8x8) {
//     const size_t rows = 8;
//     const size_t cols = 8;
//     const float A[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
//                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
//                        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
//                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
//                        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f};
//     const float B[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
//                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
//                        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
//                        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
//                        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
//                        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
//                        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
//                        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f};
//     float C_ref[rows*cols] = {0.0f};
//     float C_comp[rows*cols] = {0.0f};

//     gemm_ref(A, B, C_ref, rows, cols, rows);
//     GEMM_FUNC(A, B, C_comp, rows, cols, rows);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
// }

// #define TEST_GEMM(n, m, k) \
//     std::tuple<std::integral_constant<size_t, (n)>, std::integral_constant<size_t, (m)>, std::integral_constant<size_t, (k)>>

// using TypesSquare = testing::Types<TEST_GEMM(4U, 4U, 4U),
//                                    TEST_GEMM(8U, 8U, 8U),
//                                    TEST_GEMM(12U, 12U, 12U),
//                                    TEST_GEMM(16U, 16U, 16U),
//                                    TEST_GEMM(128U, 128U, 128U)>;

// TYPED_TEST_SUITE(GemmSquare, TypesSquare);

// TYPED_TEST(GemmSquare, Zero_ABC) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }

// TYPED_TEST(GemmSquare, Rand_AB_Zero_C) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     std::mt19937 rng;
//     rng.seed(std::random_device()());
//     std::uniform_real_distribution<ElemType> dist(0, 10);

//     std::generate(A.begin(), A.end(), [&] { return dist(rng); });
//     std::generate(B.begin(), B.end(), [&] { return dist(rng); });

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }

// TYPED_TEST(GemmSquare, Rand_ABC) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     std::mt19937 rng;
//     rng.seed(std::random_device()());
//     std::uniform_real_distribution<ElemType> dist(0, 10);

//     std::generate(A.begin(), A.end(), [&] { return dist(rng); });
//     std::generate(B.begin(), B.end(), [&] { return dist(rng); });
//     std::generate(C_ref.begin(), C_ref.end(), [&] { return dist(rng); });
//     std::copy(C_ref.begin(), C_ref.end(), C_comp.begin());

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }


// // //
// // //
// // // Non square tests
// // //
// // //

// template <typename T>
// class GemmNonSqare : public ::testing::Test
// {
// public:
//     static constexpr size_t n = std::tuple_element_t<nIndex, T>{};
//     static constexpr size_t m = std::tuple_element_t<mIndex, T>{};
//     static constexpr size_t k = std::tuple_element_t<kIndex, T>{};

//     using ElemType = float;
//     using VectorType = std::vector<ElemType>;
// };

// TEST(GemmNonSquare, SimpleTets_1x1) {
//     const size_t rows = 1;
//     const size_t cols = 1;
//     const float A[] = {1.0f};
//     const float B[] = {1.0f};
//     float C_ref[rows*cols] = {0.0f};
//     float C_comp[rows*cols] = {0.0f};

//     gemm_ref(A, B, C_ref, rows, cols, rows);
//     GEMM_FUNC(A, B, C_comp, rows, cols, rows);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
// }

// TEST(GemmNonSquare, SimpleTets_2x2) {
//     const size_t rows = 2;
//     const size_t cols = 2;
//     const float A[] = {1.0f, 2.0f,
//                        3.0f, 4.0f};
//     const float B[] = {1.0f, 2.0f,
//                        3.0f, 4.0f};
//     float C_ref[rows*cols] = {0.0f};
//     float C_comp[rows*cols] = {0.0f};

//     gemm_ref(A, B, C_ref, rows, cols, rows);
//     GEMM_FUNC(A, B, C_comp, rows, cols, rows);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
// }

// TEST(GemmNonSquare, SimpleTets_3x3) {
//     const size_t rows = 3;
//     const size_t cols = 3;
//     const float A[] = {1.0f, 2.0f, 3.0f,
//                        4.0f, 5.0f, 6.0f,
//                        7.0f, 8.0f, 9.0f};
//     const float B[] = {1.0f, 2.0f, 3.0f,
//                        4.0f, 5.0f, 6.0f,
//                        7.0f, 8.0f, 9.0f};
//     float C_ref[rows*cols] = {0.0f};
//     float C_comp[rows*cols] = {0.0f};

//     gemm_ref(A, B, C_ref, rows, cols, rows);
//     GEMM_FUNC(A, B, C_comp, rows, cols, rows);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref, C_comp, rows, cols, std::numeric_limits<float>::epsilon()));
// }

// #define TEST_GEMM(n, m, k) \
//     std::tuple<std::integral_constant<size_t, (n)>, std::integral_constant<size_t, (m)>, std::integral_constant<size_t, (k)>>

// using TypesNonSqare = testing::Types<
//                                      // Test M
//                                      TEST_GEMM(4U, 1U, 4U),
//                                      TEST_GEMM(4U, 2U, 4U),
//                                      TEST_GEMM(4U, 3U, 4U),
//                                      TEST_GEMM(4U, 5U, 4U),
//                                      TEST_GEMM(4U, 6U, 4U),
//                                      TEST_GEMM(4U, 7U, 4U),
//                                      // Test K
//                                      TEST_GEMM(4U, 4U, 1U),
//                                      TEST_GEMM(4U, 4U, 2U),
//                                      TEST_GEMM(4U, 4U, 3U),
//                                      TEST_GEMM(4U, 4U, 5U),
//                                      TEST_GEMM(4U, 4U, 6U),
//                                      TEST_GEMM(4U, 4U, 7U),
//                                      // Test M and K
//                                      TEST_GEMM(4U, 1U, 1U),
//                                      TEST_GEMM(4U, 1U, 2U),
//                                      TEST_GEMM(4U, 1U, 3U),
//                                      TEST_GEMM(4U, 1U, 5U),
//                                      TEST_GEMM(4U, 1U, 6U),
//                                      TEST_GEMM(4U, 1U, 7U),
//                                      TEST_GEMM(4U, 5U, 1U),
//                                      TEST_GEMM(4U, 5U, 2U),
//                                      TEST_GEMM(4U, 5U, 3U),
//                                      TEST_GEMM(4U, 5U, 5U),
//                                      TEST_GEMM(4U, 5U, 6U),
//                                      TEST_GEMM(4U, 5U, 7U),
//                                      TEST_GEMM(4U, 5U, 5U),
//                                      TEST_GEMM(4U, 6U, 5U),
//                                      TEST_GEMM(4U, 7U, 5U),
//                                      // Test N
//                                      TEST_GEMM(1U, 4U, 4U),
//                                      TEST_GEMM(2U, 4U, 4U),
//                                      TEST_GEMM(3U, 4U, 4U),
//                                      TEST_GEMM(5U, 4U, 4U),
//                                      TEST_GEMM(6U, 4U, 4U),
//                                      TEST_GEMM(7U, 4U, 4U),
//                                      // Test N and M
//                                      TEST_GEMM(1U, 1U, 4U),
//                                      TEST_GEMM(2U, 1U, 4U),
//                                      TEST_GEMM(3U, 1U, 4U),
//                                      TEST_GEMM(5U, 1U, 4U),
//                                      TEST_GEMM(6U, 1U, 4U),
//                                      TEST_GEMM(7U, 1U, 4U),
//                                      TEST_GEMM(1U, 5U, 4U),
//                                      TEST_GEMM(2U, 5U, 4U),
//                                      TEST_GEMM(3U, 5U, 4U),
//                                      TEST_GEMM(5U, 5U, 4U),
//                                      TEST_GEMM(6U, 5U, 4U),
//                                      TEST_GEMM(7U, 5U, 4U),
//                                      TEST_GEMM(5U, 5U, 4U),
//                                      TEST_GEMM(5U, 6U, 4U),
//                                      TEST_GEMM(5U, 7U, 4U),
//                                      // Test N and K
//                                      TEST_GEMM(1U, 4U, 1U),
//                                      TEST_GEMM(2U, 4U, 1U),
//                                      TEST_GEMM(3U, 4U, 1U),
//                                      TEST_GEMM(5U, 4U, 1U),
//                                      TEST_GEMM(6U, 4U, 1U),
//                                      TEST_GEMM(7U, 4U, 1U),
//                                      TEST_GEMM(1U, 4U, 5U),
//                                      TEST_GEMM(2U, 4U, 5U),
//                                      TEST_GEMM(3U, 4U, 5U),
//                                      TEST_GEMM(5U, 4U, 5U),
//                                      TEST_GEMM(6U, 4U, 5U),
//                                      TEST_GEMM(7U, 4U, 5U),
//                                      TEST_GEMM(5U, 4U, 5U),
//                                      TEST_GEMM(5U, 4U, 6U),
//                                      TEST_GEMM(5U, 4U, 7U),
//                                      // Test N M K
//                                      TEST_GEMM(1U, 1U, 1U),
//                                      TEST_GEMM(2U, 1U, 1U),
//                                      TEST_GEMM(3U, 1U, 1U),
//                                      TEST_GEMM(5U, 1U, 1U),
//                                      TEST_GEMM(6U, 1U, 1U),
//                                      TEST_GEMM(7U, 1U, 1U),
//                                      TEST_GEMM(1U, 5U, 1U),
//                                      TEST_GEMM(2U, 5U, 1U),
//                                      TEST_GEMM(3U, 5U, 1U),
//                                      TEST_GEMM(5U, 5U, 1U),
//                                      TEST_GEMM(6U, 5U, 1U),
//                                      TEST_GEMM(7U, 5U, 1U),
//                                      TEST_GEMM(1U, 1U, 5U),
//                                      TEST_GEMM(2U, 1U, 5U),
//                                      TEST_GEMM(3U, 1U, 5U),
//                                      TEST_GEMM(5U, 1U, 5U),
//                                      TEST_GEMM(6U, 1U, 5U),
//                                      TEST_GEMM(7U, 1U, 5U),
//                                      TEST_GEMM(1U, 5U, 5U),
//                                      TEST_GEMM(2U, 5U, 5U),
//                                      TEST_GEMM(3U, 5U, 5U),
//                                      TEST_GEMM(5U, 5U, 5U),
//                                      TEST_GEMM(6U, 5U, 5U),
//                                      TEST_GEMM(7U, 5U, 5U),
//                                      TEST_GEMM(5U, 5U, 5U),
//                                      TEST_GEMM(5U, 6U, 5U),
//                                      TEST_GEMM(5U, 7U, 5U)>;

// TYPED_TEST_SUITE(GemmNonSqare, TypesNonSqare);

// TYPED_TEST(GemmNonSqare, Zero_ABC) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }

// TYPED_TEST(GemmNonSqare, Rand_AB_Zero_C) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     std::mt19937 rng;
//     rng.seed(std::random_device()());
//     std::uniform_real_distribution<ElemType> dist(0, 100);

//     std::generate(A.begin(), A.end(), [&] { return dist(rng); });
//     std::generate(B.begin(), B.end(), [&] { return dist(rng); });

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }

// TYPED_TEST(GemmNonSqare, Rand_ABC) 
// {
//     using ElemType   = typename TestFixture::ElemType;
//     using VectorType = typename TestFixture::VectorType;

//     const size_t n = TestFixture::n;
//     const size_t m = TestFixture::m;
//     const size_t k = TestFixture::k;

//     const auto threshold = std::numeric_limits<ElemType>::epsilon() * m * 2;

//     VectorType A(n*m, 0.0f);
//     VectorType B(m*k, 0.0f);
//     VectorType C_ref(n*k, 0.0f);
//     VectorType C_comp(n*k, 0.0f);

//     std::mt19937 rng;
//     rng.seed(std::random_device()());
//     std::uniform_real_distribution<ElemType> dist(0, 100);

//     std::generate(A.begin(), A.end(), [&] { return dist(rng); });
//     std::generate(B.begin(), B.end(), [&] { return dist(rng); });
//     std::generate(C_ref.begin(), C_ref.end(), [&] { return dist(rng); });
//     std::copy(C_ref.begin(), C_ref.end(), C_comp.begin());

//     gemm_ref(A.data(), B.data(), C_ref.data(), n, m, k);
//     GEMM_FUNC(A.data(), B.data(), C_comp.data(), n, m, k);

//     ASSERT_TRUE(AssertMatricesEqual(C_ref.data(), C_comp.data(), n, k, threshold));
// }
