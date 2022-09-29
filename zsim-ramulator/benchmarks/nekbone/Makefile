# C++ compiler and flags
CC = g++
CFLAGS = -g -fopenmp -O2

all: kernels kernels_offload

OBJS = gs-setup.o

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

kernels: kernels.cpp $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS)

kernels_offload: kernels.cpp $(OBJS)
	$(CC) $(CFLAGS) -DOMP_OFFLOAD $< -o $@ $(OBJS)

format:
	clang-format -i *.cpp *.hpp

clean:
	rm -f kernels kernels_offload
