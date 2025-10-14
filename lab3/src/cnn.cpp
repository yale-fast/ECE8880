#include <cmath>
#include <tapa.h>
#include "cnn.h"

/*
void read_input(tapa::mmap<float>...,
                tapa::ostream<float>...,
                const int...
                ...);

void read_weight(tapa::mmap<float>...,
                 tapa::ostream<float>...,
                 const int...
                 ...);

void read_bias(tapa::mmap<float>...,
               tapa::ostream<float>...,
               const int...
               ...);
            
void write_output(tapa::mmap<float>...,
                  tapa::istream<float>...,
                  const int...
                  ...);

void cnn_core(tapa::istream<float>...,
              ...,
              tapa::ostream<float>...
              const int...
              ...);
...
*/

void read_input(
  tapa::mmap<float> in_img,
  tapa::ostream<float> &in_img_stream,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize
) {
  for (int i = 0; i < kNum; ++i) { // kNum kernels
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
    #pragma HLS loop_tripcount min=1 max=kNum_0
      for (int h = 0; h < kImSize; ++h) { 
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int w = 0; w < kImSize; ++w) { // each output pixel
        #pragma HLS loop_tripcount min=1 max=kImSize_0
          for (int p = 0; p < kKernel; ++p) {
          #pragma HLS loop_tripcount min=1 max=kKernel_0
            for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
            #pragma HLS loop_tripcount min=1 max=kKernel_0
            #pragma HLS PIPELINE II=1
            in_img_stream.write(in_img(j, h + p, w + q));
            }
          }
        }
      }
    }
  }
}

void read_weight(
  tapa::mmap<float> weight,
  tapa::ostream<float> &in_weight_stream,
  const int kNum,
  const int kKernel,
  const int kImSize
) {
  for (int i = 0; i < kNum; ++i) { // kNum kernels
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
    #pragma HLS loop_tripcount min=1 max=kNum_0
      for (int h = 0; h < kImSize; ++h) { 
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int w = 0; w < kImSize; ++w) { // each output pixel
        #pragma HLS loop_tripcount min=1 max=kImSize_0
          for (int p = 0; p < kKernel; ++p) {
          #pragma HLS loop_tripcount min=1 max=kKernel_0
            for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
            #pragma HLS loop_tripcount min=1 max=kKernel_0
            #pragma HLS PIPELINE II=1
              in_weight_stream.write(weight(i, j, p, q));
            }
          }
        }
      }
    }
  }
}

void read_bias(
  tapa::mmap<float> bias,
  tapa::ostream<float> &in_bias_stream,
  const int kNum,
  const int kKernel,
  const int kImSize
) {
  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
      #pragma HLS PIPELINE II=1
        in_bias_stream.write(bias[i]);
      }
    }
  }
}

void write_output(
  tapa::mmap<float> out_img,
  tapa::istream<float> &out_img_stream,
  const int kNum,
  const int kOutImSize
) {
  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kOutImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kOutImSize_0
      for (int w = 0; w < kOutImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kOutImSize_0
      #pragma HLS PIPELINE II=1
        out_img(i, h, w) = out_img_stream.read();
      }
    }
  }
}

void cnncore(
  tapa::istream<float> &in_img_stream,
  tapa::istream<float> &in_weight_stream,
  tapa::istream<float> &in_bias_stream,
  tapa::ostream<float> &out_img_stream,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize,
  const int kOutImSize) {
  static float C[kNum_0][kImSize_0][kImSize_0];

  // better notation
  const int N = kNum_0;
  const int M = kNum_0;
  const int K = kKernel_0;
  const int R = kImSize_0;
  const int C = kImSize_0;

  const int T_r = 1;
  const int T_C = 1;
  const int T_m = 64;
  const int T_n = 8;

  for (int r = 0; r < R; r += T_r) {
    for ()
  }

  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
      #pragma HLS PIPELINE II=1
        C[i][h][w] = in_bias_stream.read();
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) { // each output map
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int j = 0; j < kNum; ++j) { // each kernel kNum channels
    #pragma HLS loop_tripcount min=1 max=kNum_0
      for (int h = 0; h < kImSize; ++h) { 
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        for (int w = 0; w < kImSize; ++w) { // each output pixel
        #pragma HLS loop_tripcount min=1 max=kImSize_0
          for (int p = 0; p < kKernel; ++p) {
          #pragma HLS loop_tripcount min=1 max=kKernel_0
            for (int q = 0; q < kKernel; ++q) { // perform single kernel channel
            #pragma HLS loop_tripcount min=1 max=kKernel_0
            #pragma HLS PIPELINE II=1
              C[i][h][w] += in_weight_stream.read() * in_img_stream.read();
            }
          }
        }
      }
    }
  }
	
	// ReLU
	for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kImSize_0
      for (int w = 0; w < kImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kImSize_0
        C[i][h][w] = max(0.f, C[i][h][w]);
      }
    }
  }
	
	// Max pooling
  for (int i = 0; i < kNum; ++i) {
  #pragma HLS loop_tripcount min=1 max=kNum_0
    for (int h = 0; h < kOutImSize; ++h) {
    #pragma HLS loop_tripcount min=1 max=kOutImSize_0
      for (int w = 0; w < kOutImSize; ++w) {
      #pragma HLS loop_tripcount min=1 max=kOutImSize_0
      #pragma HLS PIPELINE II=1
        out_img_stream.write(max(
          max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
          max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1])));
      }
    }
  }
}

void CnnKernel(
  tapa::mmap<float> in_img,
  tapa::mmap<float> weight,
  tapa::mmap<float> bias,
  tapa::mmap<float> out_img,
  const int kNum,
  const int kKernel,
  const int kImSize,
  const int kInImSize,
  const int kOutImSize) {
  
  tapa::stream<float, 32> in_img_stream("q_in_image_0");
  tapa::stream<float, 32> in_weight_stream("w_in_image_0");
  tapa::stream<float, 32> in_bias_stream("b_in_image_0");
  tapa::stream<float, 32> out_img_stream("q_out_image_0");

  tapa::task()
    .invoke(read_input, in_img, in_img_stream, kNum, kKernel, kImSize, kInImSize)
    .invoke(read_weight, weight, in_weight_stream, kNum, kKernel, kImSize)
    .invoke(read_bias, bias, in_bias_stream, kNum, kKernel, kImSize)
    .invoke(write_output, out_img, out_img_stream, kNum, kOutImSize)
    .invoke(cnncore, in_img_stream, in_weight_stream, in_bias_stream, out_img_stream, kNum, kKernel, kImSize, kInImSize, kOutImSize);
}

