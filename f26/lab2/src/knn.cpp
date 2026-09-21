#include <limits>
#include <tapa.h>

#include "knn.h"

// Use README.md's data-flow diagram and channel counts to check your work.
// Stream reads consume data; mismatched read/write counts can stall simulation.
// Use .read() and .write() for stream access throughout the missing code.

void ReadImage(tapa::mmap<memory_type> input_v,
               tapa::ostream<memory_type>& output_stream) {
  for (int repeat = 0; repeat < kTestImages; ++repeat) {
    for (int i = 0; i < kTrainImagesPerClass * kImageWords; ++i) {
#pragma HLS PIPELINE II=1
      output_stream.write(input_v[i]);
    }
  }
}

// Read test images from memory and broadcast it to all ten KnnDist modules.
void ReadTestImages(tapa::mmap<memory_type> test_image,
                    tapa::ostream<memory_type>& test_stream_0,
                    tapa::ostream<memory_type>& test_stream_1,
                    tapa::ostream<memory_type>& test_stream_2,
                    tapa::ostream<memory_type>& test_stream_3,
                    tapa::ostream<memory_type>& test_stream_4,
                    tapa::ostream<memory_type>& test_stream_5,
                    tapa::ostream<memory_type>& test_stream_6,
                    tapa::ostream<memory_type>& test_stream_7,
                    tapa::ostream<memory_type>& test_stream_8,
                    tapa::ostream<memory_type>& test_stream_9) {
  for (int i = 0; i < kTestImages * kImageWords; ++i) {
#pragma HLS PIPELINE II=1
    const memory_type word = test_image[i];
    test_stream_0.write(word);
    test_stream_1.write(word);
    test_stream_2.write(word);
    test_stream_3.write(word);
    test_stream_4.write(word);
    test_stream_5.write(word);
    test_stream_6.write(word);
    test_stream_7.write(word);
    test_stream_8.write(word);
    test_stream_9.write(word);
  }
}

// One instance computes distances for all training images of one class.
// TODO: Complete the input types; each word carries kBytesPerWord image bytes
// (see memory_type in knn.h).
void KnnDist(tapa::istream<...>& train_stream,
             tapa::istream<...>& test_stream,
             tapa::ostream<uint32_t>& distance_stream) {
  for (int t = 0; t < kTestImages; ++t) {
    // TODO: Prepare this test image for comparisons with this class's training set.
    ...
    for (int tr = 0; tr < kTrainImagesPerClass; ++tr) {
      // TODO: compute a full-image distance for one training image and output to distance_stream.
      // Match the distance definition in KNN_host.
      ...
    }
  }
}

// Reduce all distances to one label per test image.
void KnnPredict(tapa::istream<uint32_t>& distance_stream_0,
                tapa::istream<uint32_t>& distance_stream_1,
                tapa::istream<uint32_t>& distance_stream_2,
                tapa::istream<uint32_t>& distance_stream_3,
                tapa::istream<uint32_t>& distance_stream_4,
                tapa::istream<uint32_t>& distance_stream_5,
                tapa::istream<uint32_t>& distance_stream_6,
                tapa::istream<uint32_t>& distance_stream_7,
                tapa::istream<uint32_t>& distance_stream_8,
                tapa::istream<uint32_t>& distance_stream_9,
                tapa::ostream<uint32_t>& prediction_stream) {
  for (int t = 0; t < kTestImages; ++t) {
    int best_label = 0;
    uint32_t best_dist = std::numeric_limits<uint32_t>::max();
    for (int tr = 0; tr < kTrainImagesPerClass; ++tr) {
      // TODO: Compare the distance from every class and
      // maintain the best class label for one test image. 
      // Equal distances must give the same label as KNN_host.
      ...
    }
    prediction_stream.write(best_label);
  }
}

void WriteLabel(tapa::istream<uint32_t>& input_stream,
                tapa::mmap<uint32_t> predict_label,
                tapa::ostream<bool>& done_stream) {
  for (int i = 0; i < kTestImages; ++i) {
#pragma HLS PIPELINE II=1
    predict_label[i] = input_stream.read();
  }
  // Send exactly one completion token after all predictions are written.
  done_stream.write(true);
}

// This loop measures cycles in RTL/hardware, not software-simulation timing.
void Timer(tapa::istream<bool>& done_stream,
           tapa::mmap<uint32_t> cycle_count) {
  uint32_t count = 0;
  while (done_stream.empty()) {
#pragma HLS PIPELINE II=1
    ++count;
  }
  done_stream.read();
  cycle_count[0] = count;
}

// The image ports use memory_type, matching the declaration in knn.h.
void KNNKernel(tapa::mmap<memory_type> train_image_0,
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
               tapa::mmap<uint32_t> cycle_count) {
  // Each suffix identifies one class. Every FIFO has kStreamDepth entries.
  // TODO: Complete all three stream groups.
  tapa::stream<..., kStreamDepth> train_stream_0("train_stream_0");
  ...

  tapa::stream<..., kStreamDepth> test_stream_0("test_stream_0");
  ...

  tapa::stream<..., kStreamDepth> distance_stream_0("distance_stream_0");
  ...

  tapa::stream<uint32_t, kStreamDepth> prediction_stream("prediction_stream");

  tapa::stream<bool, kStreamDepth> done_stream("done_stream");

  // Each invoke creates a concurrent task instance: ten readers and ten
  // distance engines feed a single predictor through the channels above.
  // TODO: Complete the task connections using the README diagram and function
  // signatures(be careful about the orders when you are connecting streams).
  tapa::task()
      .invoke(ReadImage, train_image_0, train_stream_0)
      ...
      .invoke(ReadTestImages, test_image,
              test_stream_0, ...)
      .invoke(KnnDist, ...)
      ...
      .invoke(KnnPredict,
              distance_stream_0, distance_stream_1, distance_stream_2,
              distance_stream_3, distance_stream_4, distance_stream_5,
              distance_stream_6, distance_stream_7, distance_stream_8,
              distance_stream_9, prediction_stream)
      .invoke(WriteLabel, prediction_stream, predict_label, done_stream)
      .invoke(Timer, done_stream, cycle_count);
}
