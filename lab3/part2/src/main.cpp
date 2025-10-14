#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>
#include <iostream>
#include <string>

#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(btstm, "", "path to the bitstream file, run csim if empty");
DEFINE_string(dtf, "../data", "data directory, default is ./data");

// Sequential CNN implementation
void CnnSequential(
    aligned_vector<float> & in_img,
    aligned_vector<float> & weight, 
    aligned_vector<float> & bias,
    aligned_vector<float> & out_img) {

  // Allocate memory on heap to avoid stack overflow.
  static float C[kNum][kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C[i][h][w] = bias[i];
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C[i][h][w] += weight(i, j, p, q) * in_img(j, h + p, w + q);
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = max(0.f, C[i][h][w]);
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        out_img(i, h, w) = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
      }
    }
  }
}

void LoadData(const string& data_dir, 
               aligned_vector<float> & input,
               aligned_vector<float> & weight, 
               aligned_vector<float> & bias) {
  const char kInputFile[] = "/input.bin";
  const char kWeightFile[] = "/weight.bin";
  const char kBiasFile[] = "/bias.bin";

  int input_fd = open((data_dir + kInputFile).c_str(), O_RDONLY);
  int weight_fd = open((data_dir + kWeightFile).c_str(), O_RDONLY);
  int bias_fd = open((data_dir + kBiasFile).c_str(), O_RDONLY);

  if (input_fd == -1) {
    clog << "Cannot find " << kInputFile << endl;
    exit(EXIT_FAILURE);
  }
  if (weight_fd == -1) {
    clog << "Cannot find " << kWeightFile << endl;
    exit(EXIT_FAILURE);
  }
  if (bias_fd == -1) {
    clog << "Cannot find " << kBiasFile << endl;
    exit(EXIT_FAILURE);
  }

  auto input_in = reinterpret_cast<float*>(mmap(
      nullptr, sizeof(*input.data()) * kNum * kInImSize * kInImSize, PROT_READ, MAP_SHARED, input_fd, 0));
  if (input_in == MAP_FAILED) {
    clog << "Incomplete " << kInputFile << endl;
    close(input_fd);
    exit(EXIT_FAILURE);
  }

  auto weight_in = reinterpret_cast<float*>(mmap(
      nullptr, sizeof(*weight.data()) * kNum * kNum * kKernel * kKernel, PROT_READ, MAP_SHARED, weight_fd, 0));
  if (weight_in == MAP_FAILED) {
    clog << "Incomplete " << kWeightFile << endl;
    close(weight_fd);
    exit(EXIT_FAILURE);
  }

  float* bias_in = reinterpret_cast<float*>(mmap(
      nullptr, sizeof(*bias.data()) * kNum, PROT_READ, MAP_SHARED, bias_fd, 0));
  if (bias_in == MAP_FAILED) {
    clog << "Incomplete " << kBiasFile << endl;
    close(bias_fd);
    exit(EXIT_FAILURE);
  }

  memcpy(input.data(), input_in, sizeof(*input.data()) * kNum * kInImSize * kInImSize);
  memcpy(weight.data(), weight_in, sizeof(*weight.data()) * kNum * kNum * kKernel * kKernel);
  memcpy(bias.data(), bias_in,  sizeof(*bias.data()) * kNum);
  munmap(input_in, sizeof(*input.data()) * kNum * kInImSize * kInImSize);
  munmap(weight_in, sizeof(*weight.data()) * kNum * kNum * kKernel * kKernel);
  munmap(bias_in,  sizeof(*bias.data()) * kNum);
  close(input_fd);
  close(weight_fd);
  close(bias_fd);
}

float IsError(float a, float b) {
  return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

//verify against ground truth
int Verify(const string& data_dir,
           aligned_vector<float> & out_img) {
  int error = 0;
  const char kOutputFile[] = "/output.bin";
  int fd = open((data_dir + kOutputFile).c_str(), O_RDONLY);
  if (fd == -1) {
    clog << "Cannot find " << kOutputFile << endl;
    return EXIT_FAILURE;
  }
  auto ground_truth = reinterpret_cast<float*>(mmap(
      nullptr, sizeof(*out_img.data()) * out_img.size(), PROT_READ, MAP_SHARED, fd, 0));
  if (ground_truth == MAP_FAILED) {
    clog << "Incomplete " << kOutputFile << endl;
    close(fd);
    return EXIT_FAILURE;
  }
  bool first = true;
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        if (IsError(out_img(i, h, w), ground_truth[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)])) {
          if (first) {
            clog << "First error: get " << out_img(i, h, w) << ", expecting "
                 << ground_truth[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)] << " @ i = " << i << ", h = " << h
                 << ", w = " << w << endl;
            first = false;
          }
          ++error;
        }
      }
    }
  }
  munmap(ground_truth, sizeof(*out_img.data()) * out_img.size());
  close(fd);
  return error;
}

//verify against CPU output
int Verify_againt_cpu(
  aligned_vector<float> & output_cpu,
  aligned_vector<float> & out_img) {
  int error = 0;
  bool first = true;
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        if (IsError(out_img(i, h, w), output_cpu[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)])) {
          if (first) {
            clog << "First error: get " << out_img(i, h, w) << ", expecting "
                 << output_cpu[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)] << " @ i = " << i << ", h = " << h
                 << ", w = " << w << endl;
            first = false;
          }
          ++error;
        }
      }
    }
  }
  return error;
}

bool end_with(const std::string &value, const std::string &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  clog << "CNN paprameters:" << endl
       << "  kNum = " << kNum << endl
       << "  kKernel = " << kKernel << endl
       << "  kImSize = " << kImSize << endl
       << "  kInImSize = " << kInImSize << endl
       << "  kOutImSize = " << kOutImSize << endl;

  //host data
  aligned_vector<float> h_input(kNum * kInImSize * kInImSize);
  aligned_vector<float> h_weight(kNum * kNum * kKernel * kKernel);
  aligned_vector<float> h_bias(kNum);
  aligned_vector<float> h_output(kNum * kOutImSize * kOutImSize);

  //a vector on host to store data from FPGA device
  aligned_vector<float> d_output(kNum * kOutImSize * kOutImSize);

  if (argc > 2) {
    clog << "Usage: " << argv[0] << " [data dir]\n";
    return EXIT_FAILURE;
  }

  LoadData(FLAGS_dtf, h_input, h_weight, h_bias);

  clog << "CNN computation on CPU using CnnSequential\n";
  const auto begin = steady_clock::now();
  CnnSequential(h_input, h_weight, h_bias, h_output);
  const auto end = steady_clock::now();

  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
                   / (run_time_us * 1e3);
  clog << "Time: " << run_time_us * 1e-6 << " s\n";
  clog << "Perf: " << gflops << " GFlops, CPU sequential version.\n";
  
  //run tapa kernel: 
  if (FLAGS_btstm.empty()) {
    clog << "Runing kernel in csim mode" << endl;
  } else if (end_with(FLAGS_btstm, ".xo")) {
    clog << "Runing kernel in TAPA fast cosim using file: " << FLAGS_btstm << endl;
  } else if (end_with(FLAGS_btstm, ".hw_emu.xclbin")) {
    clog << "Runing kernel in Vitis hardware emulation using file: " << FLAGS_btstm << endl;
  } else if (end_with(FLAGS_btstm, ".xclbin")) {
    clog << "Runing kernel on FPGA card using bitstream file: " << FLAGS_btstm << endl;
  } else {
    throw std::runtime_error("Unsupported bitstream file: " + FLAGS_btstm);
    return EXIT_FAILURE;
  }
  double time_taken
    = tapa::invoke(CnnKernel, FLAGS_btstm,
                   tapa::read_only_mmap<float>(h_input),//.reinterpret<float_v4>(), 
                   tapa::read_only_mmap<float>(h_weight), 
                   tapa::read_only_mmap<float>(h_bias), 
                   tapa::write_only_mmap<float>(d_output)
                  );
  time_taken *= 1e-6; // total time in mini second
  clog << "Kernel time is " << time_taken << " ms\n";
  clog << "Perf: " << (float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2 * 1e-9) / (time_taken * 1e-3) 
       << " GFlops, kernel.\n";

  //veryfy device results against cpu results
  int error = Verify_againt_cpu(
    h_output, d_output);
  if (error != 0) {
    clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
    clog << "FAIL" << endl;
    return EXIT_FAILURE;
  } else {
    clog << "PASS" << endl;
    return EXIT_SUCCESS;
  }
}
