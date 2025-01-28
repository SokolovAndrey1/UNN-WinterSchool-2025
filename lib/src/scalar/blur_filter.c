#include <scalar.h>

void blur_filter_scalar(uint8_t* src, uint8_t* dst, int width, int height, int kernel_size) {
    const int channels = 3;
    int half_kernel = kernel_size / 2;
    uint16_t div = kernel_size * kernel_size;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t sumr = 0;
            uint16_t sumg = 0;
            uint16_t sumb = 0;

            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {

                    int src_x = x + kx;
                    int src_y = y + ky;

                    // Replicate border
                    if (src_x < 0) src_x = 0;
                    if (src_x >= width) src_x = width - 1;
                    if (src_y < 0) src_y = 0;
                    if (src_y >= height) src_y = height - 1;

                    int src_offset = (src_y * width + src_x) * channels;

                    sumr = sumr + src[src_offset + 0];
                    sumg = sumg + src[src_offset + 1];
                    sumb = sumb + src[src_offset + 2];
                }
            }

            int dst_offset = (y * width + x) * channels;

            dst[dst_offset + 0] = (uint8_t)(sumr / div);
            dst[dst_offset + 1] = (uint8_t)(sumg / div);
            dst[dst_offset + 2] = (uint8_t)(sumb / div);
        }
    }
}
