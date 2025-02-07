#include "rvv.h"

size_t strlen_rvv(const char *str)
{
    size_t len = 0;
    const char *ptr = str;
    size_t vl;
    vuint8m8_t chunk;
    vbool1_t all_zero;

    while (1) {
         vl = __riscv_vsetvlmax_e8m8();
         chunk = __riscv_vle8_v_u8m8(ptr, vl); // Загрузка вектора байтов из памяти
         all_zero = __riscv_vmseq_vx_u8m8_b1(chunk, 0, vl); // Сравнение элементов вектора с нулём
         long first_zero = __riscv_vfirst_m_b1(all_zero, vl); // Поиск первого нулевого байта
         if (first_zero >= 0) {// Если нулевой байт найден
               return len + first_zero;
         }
         len += vl;
         ptr += vl;
    }
}