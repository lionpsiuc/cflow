CC   = gcc
NVCC = nvcc

CFLAGS  = -funroll-loops -march=native -O3 -std=c2x -Wall -Wextra
NVFLAGS = -O3 --use_fast_math

TARGET = assignment2

CSRCS  = assignment2.c average.c iteration.c utils.c
CUSRCS = iteration-gpu.cu
HDRS   = average.h iteration.h iteration-gpu.h utils.h

COBJS  = $(CSRCS:.c=.o)
CUOBJS = $(CUSRCS:.cu=.o)
OBJS   = $(COBJS) $(CUOBJS)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@ -lm

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(COBJS) $(CUOBJS)

.PHONY: all clean
