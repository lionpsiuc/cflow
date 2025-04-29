CC   = gcc
NVCC = nvcc

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# Create obj directories if they don't exist
$(shell mkdir -p $(OBJ_DIR)/cpu $(OBJ_DIR)/gpu)

# Compiler flags
CFLAGS = -funroll-loops -march=native -O3 -std=c11 -Wall -Wextra -I$(INC_DIR) -I.
NVFLAGS = -O3 --use_fast_math -arch=sm_86 -I$(INC_DIR) -I.

# Target executable
TARGET = assignment2

# Source files
CPU_SRCS = $(SRC_DIR)/cpu/average.c $(SRC_DIR)/cpu/iteration.c
GPU_SRCS = $(SRC_DIR)/gpu/average.cu $(SRC_DIR)/gpu/iteration.cu
COMMON_SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/utils.c

# Object files
CPU_OBJS = $(patsubst $(SRC_DIR)/cpu/%.c, $(OBJ_DIR)/cpu/%.o, $(CPU_SRCS))
GPU_OBJS = $(patsubst $(SRC_DIR)/gpu/%.cu, $(OBJ_DIR)/gpu/%.o, $(GPU_SRCS))
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

# All object files
OBJS = $(CPU_OBJS) $(GPU_OBJS) $(COMMON_OBJS)

# Default target
all: $(TARGET)

# Linking the target
$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@ -lm

# Compile CPU source files
$(OBJ_DIR)/cpu/%.o: $(SRC_DIR)/cpu/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile GPU source files
$(OBJ_DIR)/gpu/%.o: $(SRC_DIR)/gpu/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Compile common source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ_DIR)/cpu/*.o $(OBJ_DIR)/gpu/*.o $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
