# C++ compiler and flags
CC = g++
CFLAGS = -g -fopenmp -O2

all: kernels

kernels: kernels.cpp
	$(CC) $(CFLAGS) $< -o $@

kernels_offload: kernels.cpp
	$(CC) $(CFLAGS) -DOMP_OFFLOAD $< -o $@

clean:
	rm -f kernels
