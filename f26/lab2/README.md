# Lab 2-1: K-Nearest Neighbors (KNN)

First, [install the bundled course version of TAPA](../tapa/README.md). The
Makefile selects Xilinx Vivado/Vitis 2024.2 from the NRP Coder workspace.

In this lab, complete the starter kernel to classify CIFAR-10 images using
1-nearest neighbor (`K = 1`). For each test image, the completed kernel must
compute the squared Euclidean distance to each selected training image and
return the class of the closest image, matching the supplied CPU reference:

```text
distance = sum((train_byte[i] - test_byte[i])^2) over all 3072 image bytes
```

There is no square root because it would not change which image is nearest.
This baseline compares raw color values; it does not train a neural network.

## Dataset

This lab includes prepared CIFAR-10 data in [cifar-10/](cifar-10); no separate
download is required. These files use the lab-specific binary layout described
below.

The original [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
is by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton; see Alex Krizhevsky,
[*Learning Multiple Layers of Features from Tiny Images* (2009)](https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf).

| File in `cifar-10/` | Contents | Size |
| --- | --- | --- |
| `train_image_<class>.bin` (one file per class, 0 through 9) | 5000 images of that class | 15,360,000 bytes each |
| `test_image.bin` | 10000 test images | 30,720,000 bytes |
| `test_label.bin` | One little-endian `uint32_t` label per test image | 40,000 bytes |

Each image occupies 3072 consecutive `uint8_t` values (32 x 32 x 3): red,
green, then blue channel planes. Image files have no headers or embedded
labels. The training filename supplies the class label; test labels are
stored separately. These prepared files are not interchangeable with the
original CIFAR-10 batch files, which include a label byte in each record.

The CPU reference processes individual bytes and classifies test images
sequentially. The Makefile sets `TAPA_CONCURRENCY=8`
for the software-simulation task workers. The kernel accesses the same storage
as packed words of `memory_type`, `kImageWords` words per image; use the shared
layout when completing the missing interface types and distance computation.
The shared layout constants are in [src/knn.h](src/knn.h).

Set `memory_type` in [src/knn.h](src/knn.h) to the word type of your train/test
image memory interface. The default `uint32_t` moves 4 bytes per access
(768 words per image); a wider type such as `ap_uint<512>` moves 64 bytes per
access (48 words per image) and reduces the number of memory accesses.
`kImageWords` follows from `memory_type` automatically, and the host
`reinterpret<...>()` calls in [src/main.cpp](src/main.cpp) must use the same
type. With a 512-bit `memory_type`, declare `memory_type` as `ap_uint<512>`,
uncomment the `#include <ap_int.h>` line in [src/knn.h](src/knn.h), and
reinterpret the host image buffers with `reinterpret<ap_uint<512>>()`.

## Kernel data flow

The diagrams describe the required completed design. The starter supplies the
readers, output writer, and timer; you must complete the missing types, distance
and prediction logic, stream declarations, and task connections in `src`.

Ten `KnnDist` tasks compute distances in parallel, one per class. First,
`ReadTestImages` reads each test image once and broadcasts it to ten separate
streams. A stream read consumes a value, so the tasks cannot share one input
stream:

```text
test_image mmap --> ReadTestImages
                         |
                         +--> test_stream_0
                         +--> test_stream_1
                         +--> test_stream_2
                         +--> test_stream_3
                         +--> test_stream_4
                         +--> test_stream_5
                         +--> test_stream_6
                         +--> test_stream_7
                         +--> test_stream_8
                         +--> test_stream_9
```

Each test stream feeds its matching distance task below. The `_0` through
`_9` suffixes identify task instances in the diagram, not separate C++
functions: the completed `KNNKernel` must invoke `ReadImage` and `KnnDist` once
for each class.
The channel from `ReadImage_c` to `KnnDist_c` is `train_stream_c`.
Declare each stream individually, not in an array; the broadcaster and
predictor also use individual stream arguments. Keep all FIFOs, including
`prediction_stream` and `done_stream`, at `kStreamDepth = 2`, defined in
`src/knn.h`. Use `.read()` and `.write()` for stream access.

```text
train_image_0 mmap --> ReadImage_0 --+--> KnnDist_0 --> distance_stream_0 --+
test_stream_0 -----------------------+                                      |
                                                                            |
train_image_1 mmap --> ReadImage_1 --+--> KnnDist_1 --> distance_stream_1 --+
test_stream_1 -----------------------+                                      |
                                                                            |
train_image_2 mmap --> ReadImage_2 --+--> KnnDist_2 --> distance_stream_2 --+
test_stream_2 -----------------------+                                      |
                                                                            |
train_image_3 mmap --> ReadImage_3 --+--> KnnDist_3 --> distance_stream_3 --+
test_stream_3 -----------------------+                                      |
                                                                            |
train_image_4 mmap --> ReadImage_4 --+--> KnnDist_4 --> distance_stream_4 --+
test_stream_4 -----------------------+                                      |
                                                                            |
train_image_5 mmap --> ReadImage_5 --+--> KnnDist_5 --> distance_stream_5 --+
test_stream_5 -----------------------+                                      |
                                                                            |
train_image_6 mmap --> ReadImage_6 --+--> KnnDist_6 --> distance_stream_6 --+
test_stream_6 -----------------------+                                      |
                                                                            |
train_image_7 mmap --> ReadImage_7 --+--> KnnDist_7 --> distance_stream_7 --+
test_stream_7 -----------------------+                                      |
                                                                            |
train_image_8 mmap --> ReadImage_8 --+--> KnnDist_8 --> distance_stream_8 --+
test_stream_8 -----------------------+                                      |
                                                                            |
train_image_9 mmap --> ReadImage_9 --+--> KnnDist_9 --> distance_stream_9 --+
test_stream_9 -----------------------+                                      |
                                                                            v
                                                                        KnnPredict
                                                                            |
                                                                    prediction_stream
                                                                            |
                                                                            v
                                                                        WriteLabel --> predict_label mmap
                                                                            |
                                                                       done_stream
                                                                            |
                                                                            v
                                                                          Timer --> cycle_count mmap
```

The right-hand line represents ten separate distance streams entering
`KnnPredict`, not a shared FIFO. For each test image, each `KnnDist` must emit
one distance per training image in its class, in training-image order. Each
supplied `ReadImage` replays its training images for every test image.
`KnnPredict` must consume all ten distance streams and output the closest
match's class label. Its handling of equal distances must match `KNN_host`.

`WriteLabel` writes one prediction per test image and then sends one
completion token. `Timer` runs concurrently and counts until that token
arrives; its count is only intended for cosimulation and on-board timing.
The supplied counter is 32-bit and can wrap on long runs; use HLS reports,
not this counter, for the full-dataset cycle estimates from `make hlsfull`.

When reading or editing [src/knn.cpp](src/knn.cpp), check the number of
values written and read on each channel. For `T = kTestImages` test images
and `N = kTrainImagesPerClass` training images **per class**, the required
counts for one completed kernel invocation are:

| Channel | Values per stream |
| --- | --- |
| `train_stream_c` | `T * N * kImageWords` packed words |
| `test_stream_c` | `T * kImageWords` packed words |
| `distance_stream_c` | `T * N` squared distances |
| `prediction_stream` | `T` labels |
| `done_stream` | One completion token |

Keep these producer/consumer counts and ordering consistent; a missing
write or extra read can leave a task waiting indefinitely.

## Run and verify

First, set `memory_type` in `src/knn.h` and complete all `...` placeholders in
`src/main.cpp` and `src/knn.cpp`, following the TODO comments. The starter code
will not compile until these placeholders are replaced. Keep the provided CPU
reference and verification code unchanged.

Then, from the repository root, run the software simulation:

```bash
cd f26/lab2
make swsim
```

Run commands from `f26/lab2`; the default data path is `./cifar-10`.
By default, software simulation uses the first 256 training images per class
(2560 total) and the first 128 test images. [src/main.cpp](src/main.cpp) loads
the selected images, runs the CPU reference, invokes the kernel, and verifies
the result.

The host reports classification accuracy against the supplied test labels
and checks every kernel prediction against the CPU reference. `PASS` means
the predictions match, not that classification accuracy is 100%. A mismatch
prints `FAIL` and returns a nonzero exit status. To change the sample sizes:

```bash
make swsim TRAIN_IMAGE_NUM=32 TEST_IMAGE_NUM=32
```

`TRAIN_IMAGE_NUM` is the number of training images per class (1 to 5000);
`TEST_IMAGE_NUM` is the number of test images (1 to 10000). The Makefile
passes both values as compiler definitions to the host and HLS compiler.
They become `constexpr int` values in `src/knn.h`, so HLS can determine the
loop trip counts. Changing either value invalidates the previous build.
The RTL targets (`hls`, `cosim`, `hwemu`, and `knn.xo`) default to 32 training
images per class and 16 test images to keep RTL runs short. If any RTL target
is requested, these smaller defaults apply to all targets in that `make`
invocation, unless `hlsfull` is also requested: its full-dataset defaults take
precedence. Plain `make`, `make knn`, and `make swsim` use the larger software
defaults; plain `make` and `make knn` only build the executable, while
`make swsim` also runs it. Command-line overrides work in all modes and apply
only to that invocation. To compare software simulation and cosimulation on
the same workload, pass the same counts explicitly to both commands.

Start small: work grows with the product of these counts. Workload sizes are
compile-time settings, not runtime flags. After building, `./knn --skipk`
runs only the CPU reference; it does not verify the kernel or bypass the need
to complete the starter code. Use `./knn --data=/path/to/cifar-10` to read
another folder with the same prepared file layout and compiled image counts.

After a kernel change, rerun `make swsim` and check for `PASS` before
measuring RTL performance. The software-simulation cycle count is not an
FPGA performance measurement. Compare the printed CPU time and kernel
software-simulation time for host wall-clock latency; both exclude dataset
loading. Software-simulation time is not FPGA execution latency. When running
`./knn` directly, set `TAPA_CONCURRENCY=8` in the environment as well.

Build the TAPA `.xo` and run fast RTL cosimulation with the default small
workload (32 training images per class and 16 test images):

```bash
make cosim
```

For a different compile-time workload, use, for example,
`make cosim TRAIN_IMAGE_NUM=32 TEST_IMAGE_NUM=32`. This rebuilds the host and
`.xo`; an existing hardware build cannot be resized with runtime flags.

`make hls` builds only `knn.xo`. `make hwemu` remains an alias for
`make cosim`. The Makefile uses the U55c platform
`xilinx_u55c_gen3x16_xdma_3_202210_1` and a 5.00 ns HLS clock target.

To obtain Vitis HLS cycle estimates for the full dataset, run synthesis only:

```bash
make hlsfull
```

By default, this uses 5000 training images per class (50,000 total) and 10,000
test images. Run `make hlsfull` by itself for synthesis only; combining it
with a simulation target selects the full-dataset defaults for that target
too. It reuses `knn.xo`, so a subsequent `make cosim` without overrides
rebuilds with the small 32/16 defaults. HLS task cycle estimates are not
measured whole-kernel latency; concurrent tasks overlap, and memory or stream
stalls can add cycles.
