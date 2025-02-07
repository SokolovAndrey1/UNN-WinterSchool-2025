#include "rvv.h"

void fabsf_rvv(const float* src, float* dst, size_t n)
{
    vfloat32m8_t v_src, v_dst;
    vuint32m8_t v_src_int, v_dst_int;

    const uint32_t bitmask = 0x7fffffff;
    for (size_t vl; n>0; n-=vl, src+=vl, dst+=vl) {
        vl = __riscv_vsetvl_e32m8(n);  
        v_src = __riscv_vle32_v_f32m8(src, vl); 
        v_src_int = __riscv_vreinterpret_v_f32m8_u32m8(v_src); 
        v_dst_int = __riscv_vand_vx_u32m8(v_src_int, bitmask, vl); 
        v_dst = __riscv_vreinterpret_v_u32m8_f32m8(v_dst_int); 
        __riscv_vse32_v_f32m8(dst, v_dst, vl); 
    }
}
