NVCC      ?= nvcc
ARCH      ?= sm_75
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

BIN  := build/sgemm
SRCS := src/main.cu $(wildcard src/*.cuh src/kernels/*.cuh)

.PHONY: all test bench clean

all: $(BIN)

$(BIN): $(SRCS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ src/main.cu -lcublas

# Non-square shapes catch swapped M/N/K indexing that square ones hide;
# 97x131x77 exercises the bounds checks in kernels 1-3 (4-6 skip it).
test: $(BIN)
	./$(BIN) all 128 256x384x128 384x128x264 97x131x77 > /dev/null
	@echo "all kernels match cuBLAS"

bench: $(BIN)
	@mkdir -p results
	./$(BIN) all | tee results/bench.csv

clean:
	rm -rf build
