#include "rvv.h"
// LMUL1
// void fabsf_rvv(const float* src, float* dst, size_t n)
// {
//     vfloat32m1_t v_src, v_dst;


//     for (size_t vl ; n > 0; n-=vl, src+=vl, dst+=vl)
//     {
//         vl =  __riscv_vsetvl_e32m1(n);

//         v_src = __riscv_vle32_v_f32m1(src, vl);
//         v_dst = __riscv_vle32_v_f32m1(dst, vl);

//         vuint32m1_t i_v_src = __riscv_vreinterpret_v_f32m1_u32m1(v_src); 
//         i_v_src = __riscv_vand_vx_u32m1(i_v_src, 0x7fffffff, vl);
//         v_dst = __riscv_vreinterpret_v_u32m1_f32m1(i_v_src);

//         __riscv_vse32_v_f32m1(dst, v_dst, vl);
//     }
    

//     // return fabsf_scalar(src, dst, n);
// }

// LMUL 8
void fabsf_rvv(const float* src, float* dst, size_t n)
{
    vfloat32m8_t v_src, v_dst;


    for (size_t vl ; n > 0; n-=vl, src+=vl, dst+=vl)
    {
        vl =  __riscv_vsetvl_e32m8(n);

        v_src = __riscv_vle32_v_f32m8(src, vl);
        v_dst = __riscv_vle32_v_f32m8(dst, vl);

        vuint32m8_t i_v_src = __riscv_vreinterpret_v_f32m8_u32m8(v_src); 
        i_v_src = __riscv_vand_vx_u32m8(i_v_src, 0x7fffffff, vl);
        v_dst = __riscv_vreinterpret_v_u32m8_f32m8(i_v_src);

        __riscv_vse32_v_f32m8(dst, v_dst, vl);
    }
    

    // return fabsf_scalar(src, dst, n);
}
