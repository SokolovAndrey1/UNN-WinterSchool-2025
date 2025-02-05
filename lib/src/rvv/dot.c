#include "rvv.h"
#include <stdalign.h>

// redosum на каждой итерации
// float dot_rvv(const float* x, const float* y, size_t n)
// {
//     float dot = 0.0f;
//     vfloat32m8_t v_x, v_y;
//     vfloat32m1_t v_dot = __riscv_vfmv_v_f_f32m1(0, 1);

//     for (size_t vl; n > 0; n-=vl, x+=vl, y+=vl) {
//         vl = __riscv_vsetvl_e32m8(n);
//         v_x = __riscv_vle32_v_f32m8(x, vl);
//         v_y = __riscv_vle32_v_f32m8(y, vl);

//         // попарно перемножаем и складываем v_x, v_y
//         vfloat32m8_t mul_sum =__riscv_vfmv_v_f_f32m8(0, vl);
//         mul_sum = __riscv_vfmacc_vv_f32m8(mul_sum, v_x, v_y, vl);

//         // Прибавляем сумму элементов mul_sum к v_dot
//         v_dot = __riscv_vfredosum_vs_f32m8_f32m1(mul_sum, v_dot, vl);
//     }

//     // Извлекаем первый элемент из v_dot
//     dot = __riscv_vfmv_f_s_f32m1_f32(v_dot);
//     return dot;
// }

// без redosum на каждой итерации
float dot_rvv(const float *x, const float *y, size_t n)
{
    vfloat32m8_t v_x, v_y;
    size_t vl_max = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t mul_sum = __riscv_vfmv_v_f_f32m8(0, vl_max);

    for (; n >= vl_max; n -= vl_max, x += vl_max, y += vl_max)
    {
        v_x = __riscv_vle32_v_f32m8(x, vl_max);
        v_y = __riscv_vle32_v_f32m8(y, vl_max);
        // попарно перемножаем и сумму v_x, v_y аккумулируем в mul_sum
        mul_sum = __riscv_vfmacc_vv_f32m8(mul_sum, v_x, v_y, vl_max);
    }
    vfloat32m1_t v_dot = __riscv_vfmv_v_f_f32m1(0, 1);
    v_dot = __riscv_vfredosum_vs_f32m8_f32m1(mul_sum, v_dot, vl_max);

    // Обработка хвоста
    float tail_mul_sum = 0;
    for (; n > 0; n -= 1, x += 1, y += 1)
    {
        tail_mul_sum += (*x) * (*y);
    }
    
    // Извлекаем первый элемент из v_dot и складываем с суммой из хвоста. 
    return __riscv_vfmv_f_s_f32m1_f32(v_dot) + tail_mul_sum;
}