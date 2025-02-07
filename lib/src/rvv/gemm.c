#include "rvv.h"

#include <scalar.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 4

static inline void process_block_nxk(const size_t n, const size_t k, const size_t m,
                                     const float *A, const size_t lda,
                                     const float *B, const size_t ldb,
                                           float *C, const size_t ldc) {
//     const size_t blocks_m = (m / BLOCK_SIZE) * BLOCK_SIZE;
//     size_t vl = __riscv_vsetvl_e32m1(n);
//     size_t block_m;
//     size_t tail_m = m - blocks_m;
//     for (block_m = 0; block_m < blocks_m; block_m += BLOCK_SIZE) {
//         vfloat32m1_t a00 = __riscv_vlse32_v_f32m1(A + block_m, 4*lda, vl); 
//         vfloat32m1_t a01 = __riscv_vlse32_v_f32m1(A + 1 + block_m, 4*lda, vl); 
//         vfloat32m1_t a02 = __riscv_vlse32_v_f32m1(A + 2 + block_m, 4*lda, vl); 
//         vfloat32m1_t a03 = __riscv_vlse32_v_f32m1(A + 3 + block_m, 4*lda, vl); 
//         float* a = calloc(4, sizeof(float));
//         __riscv_vse32_v_f32m1(a, a00, vl);
//         for(int i = 0;  i < 4; i++)
//             printf("a00 %f ", a[i]);
//         printf("\n");
//         __riscv_vse32_v_f32m1(a, a01, vl);
//         for(int i = 0;  i < 4; i++)
//             printf("a01 %f ", a[i]);
//         printf("\n");
//         __riscv_vse32_v_f32m1(a, a02, vl);
//         for(int i = 0;  i < 4; i++)
//             printf("a02 %f ", a[i]);
//         printf("\n");
//         __riscv_vse32_v_f32m1(a, a03, vl);
//         for(int i = 0;  i < 4; i++)
//             printf("a03 %f ", a[i]);
//         printf("\n");

//         for(size_t i = 0; i < k; i++){
//             vfloat32m1_t b00 = __riscv_vfmv_v_f_f32m1(B[i + block_m], vl);
//             vfloat32m1_t b01 = __riscv_vfmv_v_f_f32m1(B[i + ldb + block_m], vl);
//             vfloat32m1_t b02 = __riscv_vfmv_v_f_f32m1(B[i + 2*ldb + block_m], vl);
//             vfloat32m1_t b03 = __riscv_vfmv_v_f_f32m1(B[i + 3*ldb + block_m], vl);
//             __riscv_vse32_v_f32m1(a, b00, vl);
//             for(int i = 0;  i < 4; i++)
//                 printf("b00 %f ", a[i]);
//             printf("\n");
//             __riscv_vse32_v_f32m1(a, b01, vl);
//             for(int i = 0;  i < 4; i++)
//                     printf("b01 %f ", a[i]);
//                 printf("\n");
//             __riscv_vse32_v_f32m1(a, b02, vl);
//              for(int i = 0;  i < 4; i++)
//                     printf("b02 %f ", a[i]);
//                 printf("\n");
//             __riscv_vse32_v_f32m1(a, b03, vl);
//             for(int i = 0;  i < 4; i++)
//                     printf("b03 %f ", a[i]);
//                 printf("\n");
//             printf("\n");
//             // vfloat32m1_t c01 = __risc_v_vlse32_v_f32m1(C + 1, 4*ldc, vl); 
//             // vfloat32m1_t c02 = __risc_v_vlse32_v_f32m1(C + 2, 4*ldc, vl); 
//             // vfloat32m1_t c03 = __risc_v_vlse32_v_f32m1(C + 3, 4*ldc, vl);

//             vfloat32m1_t c_ = __riscv_vlse32_v_f32m1(C + i, 4*ldc, vl); 
//             __riscv_vse32_v_f32m1(a, c_, vl);
//             for(int i = 0;  i < 4; i++)
//                     printf("c_ %f ", a[i]);
//                 printf("\n");
//             __riscv_vfmacc_vv_f32m1(c_, a00, b00, vl);
//             __riscv_vfmacc_vv_f32m1(c_, a01, b01, vl);
//             __riscv_vfmacc_vv_f32m1(c_, a02, b02, vl);
//             __riscv_vfmacc_vv_f32m1(c_, a03, b03, vl);
//             __riscv_vse32_v_f32m1(a, c_, vl);
//             for(int i = 0;  i < 4; i++)
//                     printf("c_ %f ", a[i]);
//                 printf("\n");
//             __riscv_vsse32_v_f32m1(C + i, 4*ldc, c_, vl); 
//             for(int i = 0;  i < 4; i++)
//                     printf("C %f ", C[i]);
//                 printf("\n");
//             printf("\n");
//             // __riscv_vsse32_v_f32m8(C + 1, c01, ldc, vl);
//             // __riscv_vsse32_v_f32m8(C + 2, c02, ldc, vl);
//             // __riscv_vsse32_v_f32m8(C + 3, c03, ldc, vl);
//         } 
//     }
    
        const float *ptr_a = A;
    const float *ptr_b = B;

    size_t blocks = (m / BLOCK_SIZE) * BLOCK_SIZE;
    size_t tail = m - blocks;
    
    size_t vl = __riscv_vsetvl_e32m1(n);

///////////////// 1
    for (size_t rem = blocks; rem > 0; rem -= BLOCK_SIZE)
    {
        vfloat32m1_t vec_a00_a10_a20_a30 = __riscv_vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), vl);
        vfloat32m1_t vec_a01_a11_a21_a31 = __riscv_vlse32_v_f32m1(ptr_a + 1, lda * sizeof(float), vl);
        vfloat32m1_t vec_a02_a12_a32_a32 = __riscv_vlse32_v_f32m1(ptr_a + 2, lda * sizeof(float), vl);
        vfloat32m1_t vec_a03_a13_a23_a33 = __riscv_vlse32_v_f32m1(ptr_a + 3, lda * sizeof(float), vl);

        for (size_t k_offset = 0; k_offset < k; k_offset += 1)
        {
            const float *ptr_b_col = ptr_b + k_offset;
            const float *ptr_c = C + k_offset;

            vfloat32m1_t vec_C_col = __riscv_vlse32_v_f32m1(ptr_c, ldc * sizeof(float), vl);

            vfloat32m1_t b00 = __riscv_vfmv_v_f_f32m1(*(ptr_b_col + 0 * ldb), vl);
            vfloat32m1_t b10 = __riscv_vfmv_v_f_f32m1(*(ptr_b_col + 1 * ldb), vl);
            vfloat32m1_t b20 = __riscv_vfmv_v_f_f32m1(*(ptr_b_col + 2 * ldb), vl);
            vfloat32m1_t b30 = __riscv_vfmv_v_f_f32m1(*(ptr_b_col + 3 * ldb), vl);

            vec_C_col = __riscv_vfmacc_vv_f32m1(vec_C_col, b00, vec_a00_a10_a20_a30, vl);
            vec_C_col = __riscv_vfmacc_vv_f32m1(vec_C_col, b10, vec_a01_a11_a21_a31, vl);
            vec_C_col = __riscv_vfmacc_vv_f32m1(vec_C_col, b20, vec_a02_a12_a32_a32, vl);
            vec_C_col = __riscv_vfmacc_vv_f32m1(vec_C_col, b30, vec_a03_a13_a23_a33, vl);

            __riscv_vsse32_v_f32m1(ptr_c, ldc * sizeof(float), vec_C_col, vl);
        }

        const size_t offset = BLOCK_SIZE * ldb;
        ptr_b += offset;
        ptr_a += BLOCK_SIZE;
    }
    
    //////////////// 2
    // size_t vl = n;
    // // ptr_a = A + blocks;
    // // ptr_b = B + blocks * ldb;

    // for(size_t i = 0; i < tail; i++){
    //     vfloat32m1_t a = __riscv_vlse32_v_f32m1(ptr_a + i, 4*lda, vl); 

    //     for(size_t j = 0; j < k; j++){
    //         vfloat32m1_t b = __riscv_vfmv_v_f_f32m1(*(ptr_b + j + i * ldb), vl);

    //         vfloat32m1_t c_ = __riscv_vlse32_v_f32m1(C + j, 4*ldc, vl); 
    //         __riscv_vfmacc_vv_f32m1(c_, a, b, vl);
    //         __riscv_vsse32_v_f32m1(C + j, 4*ldc, c_, vl); 
    //     } 
    // }
    // // ptr_a += 
    // }
 
    // // Store C (n x k) block
        for (size_t rem = tail; rem > 0; rem -= 1)
    {
        vfloat32m1_t vec_a00_a10_a20_a30 = __riscv_vlse32_v_f32m1(ptr_a + 0, lda * sizeof(float), vl);

        for (size_t k_offset = 0; k_offset < k; k_offset += 1)
        {
            const float *ptr_b_col = ptr_b + k_offset;
            const float *ptr_c = C + k_offset;

            vfloat32m1_t vec_C_col = __riscv_vlse32_v_f32m1(ptr_c, ldc * sizeof(float), vl);

            vfloat32m1_t b00 = __riscv_vfmv_v_f_f32m1(*(ptr_b_col + 0 * ldb), vl);

            vec_C_col = __riscv_vfmacc_vv_f32m1(vec_C_col, b00, vec_a00_a10_a20_a30, vl);

            __riscv_vsse32_v_f32m1(ptr_c, ldc * sizeof(float), vec_C_col, vl);
        }
        ptr_a += 1;
        ptr_b += ldb;   
    }
    }



extern void gemm_rvv(const float *A, const float *B, float *C, const size_t n, const size_t m, const size_t k)
{
    size_t blocks_n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    size_t blocks_k = (k / BLOCK_SIZE) * BLOCK_SIZE;
 
    size_t tail_n = n - blocks_n;
    size_t tail_k = k - blocks_k;
 
    size_t i;
    size_t j;
    for (i = 0; i < blocks_n; i += BLOCK_SIZE) {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE) {
            process_block_nxk(BLOCK_SIZE, BLOCK_SIZE, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
        if (tail_k != 0) {
            process_block_nxk(BLOCK_SIZE, tail_k, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
    if (tail_n != 0) {
        for (j = 0; j < blocks_k; j += BLOCK_SIZE) {
            process_block_nxk(tail_n, BLOCK_SIZE, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
        if (tail_k != 0){
            process_block_nxk(tail_n, tail_k, m, &A[i*m], m, &B[j], k, &C[(i*k) + j], k);
        }
    }
}