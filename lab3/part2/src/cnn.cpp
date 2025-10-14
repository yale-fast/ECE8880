#include <cmath>
#include <tapa.h>
#include "cnn.h"

/*
void read_input(tapa::mmap<float>...,
                tapa::ostream<float>...,
                ...);

void read_weight(tapa::mmap<float>...,
                 tapa::ostream<float>...,
                 ...);

void read_bias(tapa::mmap<float>...,
               tapa::ostream<float>...,
               ...);
            
void write_output(tapa::mmap<float>...,
                  tapa::istream<float>...,
                  ...);

void cnn_core(tapa::istream<float>...,
              ...,
              tapa::ostream<float>...
              ...);
...
*/

// void read_input(
//   tapa::mmap<float> in_img,
//   tapa::ostream<float> &in_img_stream
// ) {
//   for (int i = 0; i < kNum; ++i) { // kNum kernels
//   #pragma HLS loop_tripcount min=1 max=kNum
//     for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
//     #pragma HLS loop_tripcount min=1 max=kNum
//       for (int h = 0; h < kImSize; ++h) { 
//       #pragma HLS loop_tripcount min=1 max=kImSize
//         for (int w = 0; w < kImSize; ++w) { // each output pixel
//         #pragma HLS loop_tripcount min=1 max=kImSize
//           for (int p = 0; p < kKernel; ++p) {
//           #pragma HLS loop_tripcount min=1 max=kKernel
//             for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
//             #pragma HLS loop_tripcount min=1 max=kKernel
//             #pragma HLS PIPELINE II=1
//             in_img_stream.write(in_img(j, h + p, w + q));
//             }
//           }
//         }
//       }
//     }
//   }
// }

// void read_weight(
//   tapa::mmap<float> weight,
//   tapa::ostream<float> &in_weight_stream
// ) {
//   for (int i = 0; i < kNum; ++i) { // kNum kernels
//   #pragma HLS loop_tripcount min=1 max=kNum
//     for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
//     #pragma HLS loop_tripcount min=1 max=kNum
//       for (int h = 0; h < kImSize; ++h) { 
//       #pragma HLS loop_tripcount min=1 max=kImSize
//         for (int w = 0; w < kImSize; ++w) { // each output pixel
//         #pragma HLS loop_tripcount min=1 max=kImSize
//           for (int p = 0; p < kKernel; ++p) {
//           #pragma HLS loop_tripcount min=1 max=kKernel
//             for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
//             #pragma HLS loop_tripcount min=1 max=kKernel
//             #pragma HLS PIPELINE II=1
//               in_weight_stream.write(weight(i, j, p, q));
//             }
//           }
//         }
//       }
//     }
//   }
// }

// void read_bias(
//   tapa::mmap<float> bias,
//   tapa::ostream<float> &in_bias_stream
// ) {
//   for (int i = 0; i < kNum; ++i) {
//   #pragma HLS loop_tripcount min=1 max=kNum
//     for (int h = 0; h < kImSize; ++h) {
//     #pragma HLS loop_tripcount min=1 max=kImSize
//       for (int w = 0; w < kImSize; ++w) {
//       #pragma HLS loop_tripcount min=1 max=kImSize
//       #pragma HLS PIPELINE II=1
//         in_bias_stream.write(bias[i]);
//       }
//     }
//   }
// }

// void write_output(
//   tapa::mmap<float> out_img,
//   tapa::istream<float> &out_img_stream
// ) {
//   for (int i = 0; i < kNum; ++i) {
//   #pragma HLS loop_tripcount min=1 max=kNum_0
//     for (int h = 0; h < kOutImSize; ++h) {
//     #pragma HLS loop_tripcount min=1 max=kOutImSize
//       for (int w = 0; w < kOutImSize; ++w) {
//       #pragma HLS loop_tripcount min=1 max=kOutImSize
//       #pragma HLS PIPELINE II=1
//         out_img(i, h, w) = out_img_stream.read();
//       }
//     }
//   }
// }

void cnncore(
  tapa::mmap<float> in_img,
  tapa::mmap<float> weight,
  tapa::mmap<float> bias,
  tapa::mmap<float> out_img
  // tapa::istream<float> &in_img_stream,
  // tapa::istream<float> &in_weight_stream,
  // tapa::istream<float> &in_bias_stream,
  // tapa::ostream<float> &out_img_stream
  ) {

  // M: output fm channels kNum
  // N: input fm channels kNum
  // R: output fm n_rows kImSize
  // C: output fm n_cols kImSize
  // K: kernel width kKernel
  static float output_fm[T_m][T_r][T_c];
  static float weights[T_m][T_n][K][K];
  static float input_fm[T_n][kInImSize][kInImSize];

  // load 4-dimensional tile
  for (int r = 0; r < R; r += T_r) {
    for (int c = 0; c < C; c += T_c) {
      for (int to = 0; to < M; to += T_m) {

        // load bias
        load_bias:
        for (int too = to; too < min(to + T_m, M); too++) {
        #pragma HLS loop_tripcount min=1 max=n_T_m
          int l_o = too - to;
          for (int trr = r; trr < min(r + T_r, R); trr++) {
          #pragma HLS loop_tripcount min=1 max=n_T_r
            int l_r = trr - r;
            for (int tcc = c; tcc < min(c + T_c, C); tcc++) {
            #pragma HLS loop_tripcount min=1 max=n_T_c
            #pragma HLS PIPELINE II=1
              int l_c = tcc - c;
              output_fm[l_o][l_r][l_c] = bias[too];
            }
          }
        }

        for (int ti = 0; ti < N; ti += T_n) {

          // load input maps
          load_input:
          for (int tii = ti; tii < min(ti + T_n, N); tii++) {
          #pragma HLS loop_tripcount min=1 max=n_T_n
            int l_i = tii - ti;
            for (int trr = r; trr < min(r + T_r, R) + K - 1; trr++) {
            #pragma HLS loop_tripcount min=1 max=n_T_r
              int l_r = trr - r;
              for (int tcc = c; tcc < min(c + T_c, C) + K - 1; tcc++) {
              #pragma HLS loop_tripcount min=1 max=n_T_c
              #pragma HLS PIPELINE II=1
                int l_c = tcc - c;
                input_fm[l_i][l_r][l_c] = in_img(tii, trr, tcc);
              }
            }
          }

          // load weights
          load_weight:
          for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
              for (int too = to; too < min(to + T_m, M); too++) {
              #pragma HLS loop_tripcount min=1 max=n_T_m
                int l_o = too - to;
                for (int tii = ti; tii < min(ti + T_n, N); tii++) {
                #pragma HLS loop_tripcount min=1 max=n_T_n
                #pragma HLS PIPELINE II=1
                  int l_i = tii - ti;
                  weights[l_o][l_i][i][j] = weight(too, tii, i, j);
                }
              }
            }
          }

          // convolution
          do_convolution:
          for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
              for (int trr = r; trr < min(r + T_r, R); trr++) {
              #pragma HLS loop_tripcount min=1 max=n_T_r
                int l_r = trr - r;
                for (int tcc = c; tcc < min(c + T_c, C); tcc++) {
                #pragma HLS loop_tripcount min=1 max=n_T_c
                  int l_c = tcc - c;
                  for (int too = to; too < min(to + T_m, M); too++) {
                  #pragma HLS loop_tripcount min=1 max=n_T_m

                    int l_o = too - to;
                    for (int tii = ti; tii < min(ti + T_n, N); tii++) {
                    #pragma HLS loop_tripcount min=1 max=n_T_n
                      int l_i = tii - ti;
                      output_fm[l_o][l_r][l_c] += 
                        weights[l_o][l_i][i][j] * input_fm[l_i][l_r + i][l_c + j];
                    }
                  }
                }
              }
            }
          }
        }

        // at this stage, output_fm values are filled.
        // ReLU
        do_relu:
        for (int too = to; too < min(to + T_m, M); too++) {
        #pragma HLS loop_tripcount min=1 max=n_T_m
          int l_o = too - to;
          for (int trr = r; trr < min(r + T_r, R); trr++) {
          #pragma HLS loop_tripcount min=1 max=n_T_r
            int l_r = trr - r;
            for (int tcc = c; tcc < min(c + T_c, C); tcc++) {
            #pragma HLS loop_tripcount min=1 max=n_T_c
              int l_c = tcc - c;
              output_fm[l_o][l_r][l_c] = max(0.f, output_fm[l_o][l_r][l_c]);
            }
          }
        }
        
        // Max pooling
        do_pooling:
        for (int too = to; too < min(to + T_m, M); too++) {
        #pragma HLS loop_tripcount min=1 max=n_T_m
          int l_o = too - to;
          for (int h = r / 2; h < (min(r + T_r, R) / 2); h++) {
          #pragma HLS loop_tripcount min=1 max=n_T_r
            int l_h = h - r / 2;
            for (int w = c / 2; w < (min(c + T_c, C) / 2); w++) {
            #pragma HLS loop_tripcount min=1 max=n_T_c
            #pragma HLS PIPELINE II=1
              int l_w = w - c / 2;
              out_img(too, h, w) = max(
                max(output_fm[l_o][l_h * 2][l_w * 2    ], output_fm[l_o][l_h * 2 + 1][l_w * 2    ]),
                max(output_fm[l_o][l_h * 2][l_w * 2 + 1], output_fm[l_o][l_h * 2 + 1][l_w * 2 + 1]));
            }
          }
        }
      }
    }
  }
}

void CnnKernel(
  tapa::mmap<float> in_img,
  tapa::mmap<float> weight,
  tapa::mmap<float> bias,
  tapa::mmap<float> out_img) {

  tapa::stream<float, 32> in_img_stream("q_in_image_0");
  tapa::stream<float, 32> in_weight_stream("w_in_image_0");
  tapa::stream<float, 32> in_bias_stream("b_in_image_0");
  tapa::stream<float, 32> out_img_stream("q_out_image_0");
  tapa::task()
    // .invoke(read_input, in_img, in_img_stream)
    // .invoke(read_weight, weight, in_weight_stream)
    // .invoke(read_bias, bias, in_bias_stream)
    // .invoke(write_output, out_img, out_img_stream)
    .invoke(cnncore, in_img, weight,  bias, out_img
      //  in_img_stream, in_weight_stream, in_bias_stream, out_img_stream
      );
}