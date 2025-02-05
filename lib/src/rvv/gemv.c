#include "rvv.h"

void gemv_rvv(size_t m, size_t n, float alpha, const float* a, const float* x, float* y)
{
    const float* a_ptr = a;
    for (size_t j = 0; j < m; j++) {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vy = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i); // Обновляем vl для оставшихся элементов
            vfloat32m8_t a_vec = __riscv_vle32_v_f32m8(a_ptr + i, vl);
            vfloat32m8_t x_vec = __riscv_vle32_v_f32m8(x + i, vl);
            vy = __riscv_vfmacc_vv_f32m8(vy, a_vec, x_vec, vl);
        }
        // Умножаем результат на alpha и сохраняем в y
        vy = __riscv_vfmul_vf_f32m8(vy, alpha, vl);
        __riscv_vse32_v_f32m8(y + j, vy, vl); // Сохраняем результат в y[j]
        a_ptr += n; // Переходим к следующей строке матрицы
    }
    //__riscv_vse32_v_f32()
    // Just a placeholder
    //gemv_scalar(m, n, alpha, a, x, y);
}
