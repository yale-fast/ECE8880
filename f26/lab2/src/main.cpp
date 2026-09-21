#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "knn.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(btstm, "", "path to the bitstream file, run csim if empty");
DEFINE_string(data, "./cifar-10", "path to the CIFAR-10 binary data folder");
DEFINE_bool(skipk, false, "skip kernel execution, only host CPU if true");

// Read only the requested elements, checking the file before allocating memory.
template <typename T>
void ReadBinaryFile(const string& filename, aligned_vector<T>& data,
                    size_t count) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  const std::streamoff size = file.tellg();
  if (size < 0 || size % sizeof(T) != 0 ||
      count > static_cast<size_t>(size) / sizeof(T)) {
    throw std::runtime_error("Invalid or insufficient data in: " + filename);
  }
  data.resize(count);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
  if (!file) {
    throw std::runtime_error("Error reading file: " + filename);
  }
}

// Independent byte-wise 1-NN reference for checking the packed-word kernel.
// Visit training indices before classes, matching KnnPredict's tie-breaking.
// Provided reference: leave this function and the verification code unchanged.
void KNN_host(const std::vector<aligned_vector<uint8_t>>& train_image,
              const aligned_vector<uint8_t>& test_image,
              aligned_vector<uint32_t>& predict_label) {
  for (int t = 0; t < kTestImages; ++t) {
    int best_label = 0;
    int best_dist = std::numeric_limits<int>::max();
    for (int tr = 0; tr < kTrainImagesPerClass; ++tr) {
      for (int c = 0; c < kNumClasses; ++c) {
        int dist = 0;
        for (int i = 0; i < kImageBytes; ++i) {
          const int d = static_cast<int>(train_image[c][size_t(tr) * kImageBytes + i])
                      - static_cast<int>(test_image[size_t(t) * kImageBytes + i]);
          dist += d * d;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_label = c;
        }
      }
    }
    predict_label[t] = best_label;
  }
}

// Accuracy measures the classifier against ground truth, not kernel correctness.
void PrintAccuracy(const aligned_vector<uint32_t>& test_label,
                   const aligned_vector<uint32_t>& predict_label) {
  int correct = 0;
  for (size_t i = 0; i < predict_label.size(); ++i) {
    correct += test_label[i] == predict_label[i];
  }
  clog << "Correct: " << correct << " out of " << predict_label.size() << endl;
  clog << "Accuracy: " << 100.0 * correct / predict_label.size() << "%" << endl;
}

// Kernel correctness requires every prediction to equal the CPU reference.
int Verify(const aligned_vector<uint32_t>& output_device,
           const aligned_vector<uint32_t>& output_host) {
  int error = 0;
  for (size_t i = 0; i < output_host.size(); ++i) {
    if (output_device[i] != output_host[i]) {
      if (error < 10) {
        clog << "Mismatch at index " << i << ": device " << output_device[i]
             << ", host " << output_host[i] << endl;
      }
      ++error;
    }
  }
  return error;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  try {
    clog << "Training images per class: " << kTrainImagesPerClass
         << ", test images: " << kTestImages << endl;

    // Load the compile-time workload shared with the kernel. Images contain
    // bytes only; test labels are separate uint32_t values.
    std::vector<aligned_vector<uint8_t>> train_image(kNumClasses);
    for (int c = 0; c < kNumClasses; ++c) {
      ReadBinaryFile(FLAGS_data + "/train_image_" + std::to_string(c) + ".bin",
                     train_image[c], size_t(kTrainImagesPerClass) * kImageBytes);
    }
    aligned_vector<uint8_t> test_image;
    aligned_vector<uint32_t> test_label;
    ReadBinaryFile(FLAGS_data + "/test_image.bin", test_image,
                   size_t(kTestImages) * kImageBytes);
    ReadBinaryFile(FLAGS_data + "/test_label.bin", test_label, kTestImages);

    aligned_vector<uint32_t> label_host(kTestImages);
    // An invalid initial label makes missing kernel writes fail verification.
    aligned_vector<uint32_t> label_dev(kTestImages, kNumClasses);
    aligned_vector<uint32_t> cycle_count(1, 0);

    const auto start = steady_clock::now();
    KNN_host(train_image, test_image, label_host);
    const auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start);
    clog << "Host CPU KNN time: " << elapsed.count() << " millisecond" << endl;
    clog << "Host prediction accuracy:" << endl;
    PrintAccuracy(test_label, label_host);
    if (FLAGS_skipk) {
      return EXIT_SUCCESS;
    }

    // Empty btstm runs software simulation. Reinterpret the same image storage
    // as memory_type (see src/knn.h), matching the corresponding kernel port.
    // TODO: Complete the input arguments in the exact order declared in knn.h.
    const int64_t kernel_time_ns = tapa::invoke(KNNKernel, FLAGS_btstm,
                 tapa::read_only_mmap<uint8_t>(train_image[0]).reinterpret<...>(),
                 ...,
                 tapa::read_only_mmap<uint8_t>(test_image).reinterpret<...>(),
                 tapa::write_only_mmap<uint32_t>(label_dev),
                 tapa::write_only_mmap<uint32_t>(cycle_count));

    // For software simulation, TAPA returns elapsed wall-clock nanoseconds.
    clog << "Kernel " << (FLAGS_btstm.empty() ? "software simulation" : "execution")
         << " time: " << kernel_time_ns / 1e6 << " millisecond" << endl;
    clog << "Kernel cycle count: " << cycle_count[0]
         << " (only meaningful in cosim and on board)" << endl;
    clog << "Kernel prediction accuracy:" << endl;
    PrintAccuracy(test_label, label_dev);

    const int error = Verify(label_dev, label_host);
    if (error != 0) {
      clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
      clog << "FAIL" << endl;
      return EXIT_FAILURE;
    }
    clog << "PASS" << endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    clog << "Error: " << error.what() << endl;
    return EXIT_FAILURE;
  }
}
