ROCM_PATH?= $(wildcard /opt/rocm)
HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc
INCLUDE_DIRS=-I$(HIP_PATH)/include -I$(ROCM_PATH)/include
LD_FLAGS=-L$(ROCM_PATH)/lib -L$(ROCM_PATH)/opencl/lib/x86_64 -lMIOpen -lOpenCL -lmiopengemm -lhipblas-hcc -lrocblas-hcc
TARGET=--amdgpu-target=gfx900

HIPCC_FLAGS=-g $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)

all: alexnet resnet benchmark_wino layerwise

HEADERS=function.hpp layers.hpp miopen.hpp multi_layers.hpp tensor.hpp utils.hpp

benchmark: all
	./benchmark_wino && ./layerwise && ./alexnet

alexnet: alexnet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) alexnet.cpp $(LD_FLAGS) -o $@

resnet: resnet.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) resnet.cpp $(LD_FLAGS) -o $@

benchmark_wino: benchmark_wino.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) benchmark_wino.cpp $(LD_FLAGS) -o $@

layerwise: layerwise.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) layerwise.cpp $(LD_FLAGS) -o $@

#segfault: conv_segfault.cpp
#	$(HIPCC) $(HIPCC_FLAGS) conv_segfault.cpp $(LD_FLAGS) -o $@

clean:
	rm -f *.o *.out benchmark segfault alexnet resnet benchmark_wino layerwise
