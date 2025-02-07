#include "rvv.h"

void fabsf_rvv(const float* src, float* dst, size_t n)
{
    vfloat32m8_t v_src, v_dst;
    vuint32m8_t v_src_int, v_dst_int;
    //маска для зануления первого бита
    const uint32_t mask = 0x7fffffff;
    for(size_t vl; n > 0; n -= vl, src += vl, dst += vl) {
        //устанавливаем длину вектора 
        vl = __riscv_vsetvl_e32m8(n);
        //загружаем float вектор из src длиной vl
        v_src = __riscv_vle32_v_f32m8(src, vl);
        //битовые операции есть только для целочисл типов (?) -> кастим к uint32
        v_src_int = __riscv_vreinterpret_v_f32m8_u32m8(v_src);
        //v_src_int & mask
        v_dst_int = __riscv_vand_vx_u32m8(v_src_int, mask, vl);
        //кастим обратно в float
        v_dst = __riscv_vreinterpret_v_u32m8_f32m8(v_dst_int);
        //сохраняем результат в dst
        __riscv_vse32_v_f32m8(dst, v_dst, vl);
    }
}