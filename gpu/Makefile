CC = /usr/x86_64-pc-linux-gnu/gcc-bin/4.9.3/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /opt/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

CFLAGS = -Wall -Wextra -O3 -I$(CUDA_INC_PATH)

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 \
    		   -gencode arch=compute_30,code=sm_30 \
		   -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc \
		   --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/4.9.3 \
		   -D_FORCE_INLINES

all: mhd

mhd: numerics_gpu.o mhd.o
	$(CC) -o mhd numerics_gpu.o mhd.o -L$(CUDA_LIB_PATH) -lcudart -lm

tests: numerics_gpu.o tests.o
	$(CC) $(CFLAGS) -o tests numerics_gpu.o tests.o

numerics_gpu.o: numerics_gpu.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	    -I$(CUDA_INC_PATH) -o $@ -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o tests mhd

.PHONY: all clean

