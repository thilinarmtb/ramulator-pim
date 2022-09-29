# C++ compiler and flags
CC = g++
CFLAGS = -g -fopenmp -O2

all: kernels kernels_offload

kernels: kernels.cpp
	$(CC) $(CFLAGS) $< -o $@

kernels_offload: kernels.cpp
	$(CC) $(CFLAGS) -DOMP_OFFLOAD $< -o $@

format:
	clang-format -i *.cpp *.hpp

clean:
	rm -f kernels kernels_offload
