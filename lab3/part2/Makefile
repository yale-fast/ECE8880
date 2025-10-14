INC_XCL := 
#-I /opt/xilinx/xrt/include/
GXX_FLAGS := -w -O2 -std=c++17
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
SRC := ./src

Platform := xilinx_u55c_gen3x16_xdma_3_202210_1

.DEFAULT_GOAL := cnn

cnn.o: $(SRC)/cnn.cpp
	tapa g++ -- $(GXX_FLAGS) -c $^ $(INC_XCL)

main.o: $(SRC)/main.cpp
	tapa g++ -- $(GXX_FLAGS) -c $^ $(INC_XCL)

cnn: cnn.o main.o
	tapa g++ -- $(GXX_FLAGS) -o $@ $^ $(INC_XCL) $(LIB)

swsim: cnn
	./cnn ../data

hls: $(SRC)/cnn.cpp
	tapa compile --top CnnKernel \
	--platform xilinx_u55c_gen3x16_xdma_3_202210_1 \
	--clock-period 3.33 \
	-f $^ \
	-o cnn.xo

hwemu: cnn.xo
	./cnn --btstm=./cnn.xo 

clean:
	rm *.o cnn

cleanall:
	rm -rf work.out
	rm *.o cnn cnn.xo
