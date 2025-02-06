#include "rvv.h"

// LMUL 2-8
// void alpha_compositing_rvv(uint8_t *foreground, uint8_t *background, float alpha, uint8_t *dst, int width, int height, int channels)
// {
//     int arr_size = width * height * channels;
//     for (size_t vl; arr_size > 0; arr_size -= vl, foreground += vl, background += vl, dst += vl)
//     {
//         vl = __riscv_vsetvl_e32m8(arr_size);
//         vuint16m4_t v_u16_foreground = __riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(foreground, vl), vl);
//         vuint16m4_t v_u16_background = __riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(background, vl), vl);

//         vfloat32m8_t v_f32_foreground = __riscv_vfwcvt_f_xu_v_f32m8(v_u16_foreground, vl);
//         vfloat32m8_t v_f32_background = __riscv_vfwcvt_f_xu_v_f32m8(v_u16_background, vl);

//         v_f32_foreground = __riscv_vfmul_vf_f32m8(v_f32_foreground, alpha, vl);
//         vfloat32m8_t v_f32_dst = __riscv_vfmacc_vf_f32m8(v_f32_foreground, 1.0f - alpha, v_f32_background, vl);

//         vuint16m4_t v_u16_dst = __riscv_vfncvt_xu_f_w_u16m4(v_f32_dst, vl);
//         __riscv_vse8_v_u8m2(dst, __riscv_vncvt_x_x_w_u8m2(v_u16_dst, vl), vl); // v_u16_dst -> v_u8_dst and save in dst
//     }
// }

// LMUL 1-4
void alpha_compositing_rvv(uint8_t *foreground, uint8_t *background, float alpha, uint8_t *dst, int width, int height, int channels)
{
    int arr_size = width * height * channels;
    for (size_t vl; arr_size > 0; arr_size -= vl, foreground += vl, background += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m4(arr_size);
        vuint16m2_t v_u16_foreground = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(foreground, vl), vl);
        vuint16m2_t v_u16_background = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(background, vl), vl);

        vfloat32m4_t v_f32_foreground = __riscv_vfwcvt_f_xu_v_f32m4(v_u16_foreground, vl);
        vfloat32m4_t v_f32_background = __riscv_vfwcvt_f_xu_v_f32m4(v_u16_background, vl);

        v_f32_foreground = __riscv_vfmul_vf_f32m4(v_f32_foreground, alpha, vl);
        vfloat32m4_t v_f32_dst = __riscv_vfmacc_vf_f32m4(v_f32_foreground, 1.0f - alpha, v_f32_background, vl);

        vuint16m2_t v_u16_dst = __riscv_vfncvt_xu_f_w_u16m2(v_f32_dst, vl);
        __riscv_vse8_v_u8m1(dst, __riscv_vncvt_x_x_w_u8m1(v_u16_dst, vl), vl); // v_u16_dst -> v_u8_dst and save in dst
    }
}
