#include "rvv.h"

#include <riscv_vector.h>

void gemv_rvv(size_t m, size_t n, float alpha, const float* a, const float* x, float* y) {
    const float* a_ptr = a;
    for (size_t j = 0; j < m; j++) {
        size_t vl;
        vfloat32m8_t vy = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvl_e32m8(n));

        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i); // Обновляем vl для оставшихся элементов
            vfloat32m8_t a_vec = __riscv_vle32_v_f32m8(a_ptr + i, vl); // Загрузка A[j][i...i+vl]
            vfloat32m8_t x_vec = __riscv_vle32_v_f32m8(x + i, vl); // Загрузка x[i...i+vl]
            vy = __riscv_vfmacc_vv_f32m8(vy, a_vec, x_vec, vl); // vy += a_vec * x_vec
        }

        vfloat32m1_t vred = __riscv_vfmv_v_f_f32m1(0.0f, 1); 
        vred = __riscv_vfredusum_vs_f32m8_f32m1(vy, vred, n); //Редуцирующая сумма
        float sum = __riscv_vfmv_f_s_f32m1_f32(vred); //Извлечение скаляра

        sum *= alpha; // Умножение на alpha
        y[j] += sum;   // Сохранение результата в y[j]

        a_ptr += n; // Переход к следующей строке матрицы
    }
}
