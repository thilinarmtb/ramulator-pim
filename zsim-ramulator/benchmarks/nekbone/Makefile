# C++ compiler and flags
CC = g++
CFLAGS = -g -fopenmp -O2
OMP_OFFLOAD = 0

PP =
ifneq ($(OMP_OFFLOAD), 0)
  PP += -DOMP_OFFLOAD
endif

all: kernels

OBJS = gs.o ax.o

%.o : %.cpp
	$(CC) $(CFLAGS) $(PP) -c $< -o $@

kernels: kernels.cpp $(OBJS)
	$(CC) $(CFLAGS) $(PP) $< -o $@ $(OBJS)

format:
	clang-format -i *.cpp *.hpp

clean:
	rm -f kernels kernels_offload *.o
