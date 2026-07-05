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

CC := /usr/bin/gcc
CXX := /usr/bin/g++

ifeq ($(NVCC_DEBUG),1)
  NVCC_FLAGS := -g -G -O0 -std=c++17
  CXXFLAGS := -g3 -O0 -std=c++17
else
  NVCC_FLAGS := -O3 -std=c++17
  CXXFLAGS := -O3 -std=c++17
endif

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
	rm -f large_div

kepler:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_35 large_div.cu -o large_div -lgmp

maxwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_50 large_div.cu -o large_div -lgmp

pascal:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_60 large_div.cu -o large_div -lgmp

volta:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_70 large_div.cu -o large_div -lgmp

turing:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_75 large_div.cu -o large_div -lgmp

ampere:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_80 large_div.cu -o large_div -lgmp

ada:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_89 large_div.cu -o large_div -lgmp

blackwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../../include -arch=sm_120 large_div.cu -o large_div -lgmp
