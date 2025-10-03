# Makefile for nnfs_cublas project

# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -g -O0 -Wall -std=c++17

# Source files
SRCS = $(wildcard *.cpp)
CU_SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

# Executable name
TARGET = nnfs_cublas

# CUDA libraries
CUDA_LIBS = -lcublas -lcudart

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(CUDA_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean