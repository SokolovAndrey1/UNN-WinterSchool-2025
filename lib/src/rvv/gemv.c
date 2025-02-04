#include "rvv.h"

void gemv_rvv(size_t m, size_t n, float alpha, const float* a, const float* x, float* y)
{
    const float* a_ptr = a;
    vfloat32m8_t vy;
    for (size_t j = 0; j < m; j++) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t temp = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (size_t i = 0; i < n; i += vl) {
            vfloat32m8_t a_vec = __riscv_vle32_v_f32m8(a_ptr + i, vl);
            vfloat32m8_t x_vec = __riscv_vle32_v_f32m8(x + i, vl);

            temp = __riscv_vfmacc_vv_f32m8(temp, a_vec, x_vec, vl);
        }
        //temp = __riscv_vfmv_f_s_f32m8_f32(acc);
        vy = __riscv_vfmul_vf_f32m8(temp, alpha, vl);
        __riscv_vse32_v_f32m8(y, vy, vl); 
        a_ptr += n;
    }
    //__riscv_vse32_v_f32()
    // Just a placeholder
    //gemv_scalar(m, n, alpha, a, x, y);
}
