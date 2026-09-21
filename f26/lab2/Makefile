XILINX_VERSION := 2024.2
XILINX_ROOT := /tools/Xilinx
export XILINX_VITIS := $(XILINX_ROOT)/Vitis/$(XILINX_VERSION)
export XILINX_HLS := $(XILINX_ROOT)/Vitis_HLS/$(XILINX_VERSION)
export XILINX_VIVADO := $(XILINX_ROOT)/Vivado/$(XILINX_VERSION)
export PLATFORM_REPO_PATHS := /opt/xilinx/platforms
export PATH := $(HOME)/.rapidstream-tapa/usr/bin:$(XILINX_VITIS)/bin:$(XILINX_HLS)/bin:$(XILINX_VIVADO)/bin:$(PATH)

# Keep RTL workloads small; software simulation uses a larger sample.
# hlsfull uses all 50,000 training and 10,000 test images for cycle estimates.
# All targets in one make invocation share the same compile-time counts.
ifneq ($(filter hlsfull,$(MAKECMDGOALS)),)
TRAIN_IMAGE_NUM ?= 5000
TEST_IMAGE_NUM ?= 10000
else ifneq ($(filter hls cosim hwemu knn.xo,$(MAKECMDGOALS)),)
TRAIN_IMAGE_NUM ?= 32
TEST_IMAGE_NUM ?= 16
else
TRAIN_IMAGE_NUM ?= 256
TEST_IMAGE_NUM ?= 128
endif
DEFINES := -DTRAIN_IMAGE_NUM=$(TRAIN_IMAGE_NUM) -DTEST_IMAGE_NUM=$(TEST_IMAGE_NUM)
GXX_FLAGS := -w -O2 -std=c++17 -I$(XILINX_VITIS)/include $(DEFINES)
# Use eight TAPA task workers during software simulation.
export TAPA_CONCURRENCY := 8
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
SRC := ./src

.DEFAULT_GOAL := knn
.PHONY: swsim hls hlsfull cosim hwemu clean cleanall FORCE

# Keep the timestamp unchanged unless the configuration changes. Both the host
# and hardware then rebuild, including when switching back to previous counts.
.build-config: FORCE
	@printf '%s\n' '$(DEFINES)' > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

FORCE:

knn.o: $(SRC)/knn.cpp $(SRC)/knn.h Makefile .build-config
	tapa g++ -- $(GXX_FLAGS) -c $<

main.o: $(SRC)/main.cpp $(SRC)/knn.h Makefile .build-config
	tapa g++ -- $(GXX_FLAGS) -c $<

knn: knn.o main.o
	tapa g++ -- $(GXX_FLAGS) -o $@ $^ $(LIB)

swsim: knn
	./knn

knn.xo: $(SRC)/knn.cpp $(SRC)/knn.h Makefile .build-config
	tapa compile --top KNNKernel \
	--platform xilinx_u55c_gen3x16_xdma_3_202210_1 \
	--clock-period 5.00 \
	$(foreach define,$(DEFINES),-c $(define)) \
	-f $< \
	-o $@

hls hlsfull: knn.xo

cosim: knn knn.xo
	env -u XILINX_XRT ./knn --btstm=./knn.xo

hwemu: cosim

clean:
	rm -f *.o knn

cleanall: clean
	rm -f knn.xo .build-config .build-config.tmp
	rm -rf work.out
