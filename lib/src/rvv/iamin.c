#include "rvv.h"
#include <float.h>

// LMUL - 8
int iamin_rvv(size_t n, const float *x)
{
    int min_idx = 0;
    float minf = FLT_MAX;

    for (size_t vl, i = 0; i < n; i += vl)
    {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(&x[i], vl);
        vfloat32m8_t v_abs_x = __riscv_vfabs_v_f32m8(v_x, vl);

        vfloat32m1_t v_cur_min = __riscv_vfmv_v_f_f32m1(FLT_MAX, vl);
        v_cur_min = __riscv_vfredmin_vs_f32m8_f32m1(v_abs_x, v_cur_min, vl);
        float cur_min = __riscv_vfmv_f_s_f32m1_f32(v_cur_min);

        if (cur_min < minf)
        {
            minf = cur_min;

            vbool4_t mask = __riscv_vmfeq_vf_f32m8_b4(v_abs_x, minf, vl);
            int cur_min_idx = __riscv_vfirst_m_b4(mask, vl);

            if (cur_min_idx != -1)
            {
                min_idx = cur_min_idx + i;
            }
        }
    }
    return min_idx;
}

// LMUL - 1
// int iamin_rvv(size_t n, const float *x)
// {
//     const float *arr = x;
//     size_t i = 0, min_index = 0;
//     float min_val = FLT_MAX;
//     size_t vl;
//     for (; i < n; i += vl)
//     {
//         vl = __riscv_vsetvl_e32m1(n - i);

//         vfloat32m1_t vdata = __riscv_vle32_v_f32m1(&x[i], vl);
//         vfloat32m1_t vabs_data = __riscv_vfabs_v_f32m1(vdata, vl);

//         float local_min = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vabs_data, vabs_data, vl));

//         if (local_min < min_val)
//         {
//             min_val = local_min;

//             vbool32_t mask = __riscv_vmfeq_vf_f32m1_b32(vabs_data, min_val, vl);
//             int local_index = __riscv_vfirst_m_b32(mask, vl);

//             if (local_index != -1)
//             {
//                 min_index = i + local_index;
//             }
//         }
//     }
//     return min_index;
// }