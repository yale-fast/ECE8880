#ifndef KNN_H_
#define KNN_H_

#include <cstdint>
#include <tapa.h>

// Uncomment if memory_type is an ap_uint<> width (the Makefile already adds
// the Vitis include path that provides this header):
// #include <ap_int.h>

// Student starter: every `...` in main.cpp and knn.cpp marks missing code (a
// type, arguments, or statements), and memory_type in this file must be set to
// your memory word type. Replace all placeholders before building; they are
// not valid C++ here. See the TODO comments for requirements.

// Shared image layout for the host and kernel: 3072 bytes, or kImageWords
// packed words of memory_type. Each byte is one color-channel value.
// TODO: Set memory_type to the word type of your train/test image memory
// interface. The default uint32_t moves 4 bytes per access; a wider type
// (e.g., ap_uint<512>, 64 bytes per access) reduces the number of memory
// accesses. kImageWords and every image port, image stream, and host
// reinterpret<...>() in main.cpp must stay consistent with this type.
using memory_type = uint32_t;
constexpr int kNumClasses = 10;
constexpr int kImageBytes = 32 * 32 * 3;
constexpr int kBytesPerWord = sizeof(memory_type);
constexpr int kImageWords = kImageBytes / kBytesPerWord;
constexpr int kStreamDepth = 2;

// The Makefile supplies the same -D definitions to the host and HLS compiler.
// Use these constants directly in task loops so HLS knows their trip counts.
// Training images are counted per class; test images are counted in total.
constexpr int kTrainImagesPerClass = TRAIN_IMAGE_NUM;
constexpr int kTestImages = TEST_IMAGE_NUM;
static_assert(kTrainImagesPerClass > 0 && kTrainImagesPerClass <= 5000,
              "TRAIN_IMAGE_NUM must be between 1 and 5000");
static_assert(kTestImages > 0 && kTestImages <= 10000,
              "TEST_IMAGE_NUM must be between 1 and 10000");

// Training mmap c contains only class c. The kernel writes one label per test
// image, using kTrainImagesPerClass training images from each class.
// The image ports use memory_type; keep the definition in knn.cpp and the
// host's mmap views in main.cpp consistent with it.
void KNNKernel(
    tapa::mmap<memory_type> train_image_0,
    tapa::mmap<memory_type> train_image_1,
    tapa::mmap<memory_type> train_image_2,
    tapa::mmap<memory_type> train_image_3,
    tapa::mmap<memory_type> train_image_4,
    tapa::mmap<memory_type> train_image_5,
    tapa::mmap<memory_type> train_image_6,
    tapa::mmap<memory_type> train_image_7,
    tapa::mmap<memory_type> train_image_8,
    tapa::mmap<memory_type> train_image_9,
    tapa::mmap<memory_type> test_image,
    tapa::mmap<uint32_t> predict_label,
    tapa::mmap<uint32_t> cycle_count);

#endif
