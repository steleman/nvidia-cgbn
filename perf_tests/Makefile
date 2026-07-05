.PHONY: pick clean gmp-run gmp-numactl-run xmp-run kepler maxwell pascal volta turing ampere ada blackwell

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
	rm -f xmp_tester gmp_tester

gmp-run: gmp_tester
	@./gmp_tester

gmp-numactl-run: gmp_tester
	numactl --cpunodebind=0 ./gmp_tester

xmp-run: xmp_tester
	@./xmp_tester

gmp_tester:
	$(CXX) $(CXXFLAGS) $(INC) $(LIB) gmp_tester.cc -o gmp_tester -lgmp -fopenmp

xmp_tester: 
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
	@exit 0


kepler:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_35 xmp_tester.cu -o xmp_tester -lgmp

maxwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_50 xmp_tester.cu -o xmp_tester -lgmp

pascal:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_60 xmp_tester.cu -o xmp_tester -lgmp

volta:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_70 xmp_tester.cu -o xmp_tester -lgmp

turing:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_75 xmp_tester.cu -o xmp_tester -lgmp

ampere:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_80 xmp_tester.cu -o xmp_tester -lgmp

ada:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_89 xmp_tester.cu -o xmp_tester -lgmp

blackwell:
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I../include -arch=sm_120 xmp_tester.cu -o xmp_tester -lgmp

