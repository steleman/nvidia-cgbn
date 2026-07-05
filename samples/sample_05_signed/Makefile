ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

CUDA_VERSION ?= 13.2
CUDA_HOME := /usr/local/cuda-$(CUDA_VERSION)
NVCC := $(CUDA_HOME)/bin/nvcc

ifeq ($(NVCC_DEBUG),1)
  NVCC_FLAGS := -g -G -O0 -std=c++17
  CXXFLAGS := -g3 -O0 -std=c++17
else
  NVCC_FLAGS := -O3 -std=c++17
  CXXFLAGS := -O3 -std=c++17
endif

CC := /usr/bin/gcc
CXX := /usr/bin/g++

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo "   make ada"
	@echo "   make blackwell"
	@echo

clean:
	rm -f signed

kepler:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_35 signed.cu -o signed -lgmp

maxwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_50 signed.cu -o signed -lgmp

pascal:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_60 signed.cu -o signed -lgmp

volta:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_70 signed.cu -o signed -lgmp

turing:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_75 signed.cu -o signed -lgmp

ampere:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_80 signed.cu -o signed -lgmp

ada:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_89 signed.cu -o signed -lgmp

blackwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_120 signed.cu -o signed -lgmp
