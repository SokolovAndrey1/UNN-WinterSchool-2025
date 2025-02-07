#include "rvv.h"


void symv_rvv(size_t n, float alpha, const float* a, const float* x, float* y) {
    const size_t inc_x = 1;
    const size_t inc_y = 1;
    const size_t lda = n;
    const size_t offset = n;

    size_t jx = 0;
    size_t jy = 0;

    for (size_t j = 0; j < offset; j++) {
        float temp1 = alpha * x[jx];
        //float temp2 = 0.0f;
        vfloat32m8_t temp2 = __riscv_vfmv_v_f_f32m8(0.0f, n - j);
        size_t jlda = j * lda;
        y[jy] += temp1 * a[jlda + j];

        size_t iy = jy;
        size_t ix = jx;

        size_t vl;

        ix++;
        iy++;
        
        for (size_t i = j + 1; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i); 


            vfloat32m8_t a_vec =  __riscv_vle32_v_f32m8(&a[jlda + i], vl);
            vfloat32m8_t x_vec =  __riscv_vle32_v_f32m8(&x[ix], vl);
            vfloat32m8_t y_vec =  __riscv_vle32_v_f32m8(&y[iy], vl);

            
            y_vec =  __riscv_vfmacc_vf_f32m8(y_vec, temp1, a_vec, vl);
            temp2 =  __riscv_vfmacc_vv_f32m8(temp2, a_vec, x_vec, vl); 
            //temp2 += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(a_vec, x_vec, vl), 
              //                                                         __riscv_vfmv_v_f_f32m1(0.0, vl), vl)); 

             __riscv_vse32_v_f32m8(&y[iy], y_vec, vl);

            ix += vl * inc_x;
            iy += vl * inc_y;

        }
    
    
        y[jy] += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(
                                (temp2), __riscv_vfmv_v_f_f32m1(0.0, n - j), n - j)) * alpha;
        //y[jy] += alpha * temp2;
        
        jx += inc_x;
        jy += inc_y;
    }
}