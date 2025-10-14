#ifndef CNN_H_
#define CNN_H_

#include <tapa.h>
#include <hls_vector.h>

using float_v4 = hls::vector<float, 4>;

#define weight(i, j, p, q) \
    weight[(i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + \
    (p) * kKernel + (q)]
#define in_img(j, h, w) \
    in_img[(j) * kInImSize * kInImSize + (h) * kInImSize + (w)]
#define out_img(i, h, w) \
    out_img[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)]

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

//parameters for the default/max setting
// constexpr int kNum = 256;       // chnannel number
// constexpr int kKernel = 5;      // knernel size
// constexpr int kImSize = 224;    //image size (after conv)

//parameters for the small setting
//if you have tiled on one dimention, please set the corresponding parameter to be a multiple of the tiling factor
//for example, if you have tiled kNum by 16, please set kNum to be 32

constexpr int kNum = 32;       // chnannel number
constexpr int kKernel = 3;     // knernel size
constexpr int kImSize = 50;    //image size (after conv)


// better notation
const int N = kNum;
const int M = kNum;
const int K = kKernel;
const int R = kImSize;
const int C = kImSize;

const int T_r = 2;
const int T_c = 2;
const int T_m = 8;
const int T_n = 8;

constexpr int kInImSize = kImSize + kKernel - 1;  //input image size
constexpr int kOutImSize = kImSize / 2; //output image size (after maxpool)
constexpr int n_T_r = R / T_r;   // num of output row tiles
constexpr int n_T_c = C / T_c;   // num of output column tiles
constexpr int n_T_m = M / T_m;   // num of output channel tiles
constexpr int n_T_n = N / T_n;   // num of input channel tiles

void CnnKernel(
    tapa::mmap<float> in_img,
    tapa::mmap<float> weight,
    tapa::mmap<float> bias,
    tapa::mmap<float> out_img);

#endif
