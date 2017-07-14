ROCM_PATH?= $(wildcard /opt/rocm)
HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc
INCLUDE_DIRS=-I$(HIP_PATH)/include -I$(ROCM_PATH)/include
LD_FLAGS=-L$(ROCM_PATH)/lib -L$(ROCM_PATH)/opencl/lib/x86_64 -lMIOpen -lOpenCL -lmiopengemm -lhipblas-hcc -lrocblas-hcc
TARGET=--amdgpu-target=gfx900

HIPCC_FLAGS=-g $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)

all: benchmark segfault

benchmark: main.cpp
	$(HIPCC) $(HIPCC_FLAGS) main.cpp $(LD_FLAGS) -o $@

segfault: conv_segfault.cpp
	$(HIPCC) $(HIPCC_FLAGS) conv_segfault.cpp $(LD_FLAGS) -o $@

clean:
	rm -f *.o *.out benchmark segfault
