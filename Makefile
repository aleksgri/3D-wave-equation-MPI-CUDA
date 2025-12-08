NVCC = nvcc
HOST_COMP = mpic++

ARCH = sm_60

CXXFLAGS = -std=c++11 -I/usr/local/cuda/include -O3
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
NVCCFLAGS = -arch=$(ARCH) -Xptxas -v

TARGET = cuda_sol

CPP_SRCS = cuda_sol.cpp
CU_SRCS = cuda_sol_kernels.cu


CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(HOST_COMP) $(CU_OBJS) $(CPP_OBJS) -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(HOST_COMP) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f $(CU_OBJS) $(CPP_OBJS)