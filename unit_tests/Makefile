ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif
ifdef GTEST_HOME
  GTEST_DIR := $(GTEST_HOME)
endif
ifndef GTEST_HOME
  GTEST_DIR := ../gtest
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

# "<arch>-large" targets build/run ONLY the maximum size (256K bits) with a small
# per-test instance count -- a quick correctness check at the top of the supported
# range (a full-count run at this size would take many hours).  Override the
# instance cap with e.g. LARGE_COUNT=16.
LARGE_COUNT ?= 8
LARGE := -DLARGE_TEST -DLARGE_TEST_COUNT=$(LARGE_COUNT)

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
	@echo "Append -large (e.g. make blackwell-large) to build/run ONLY the"
	@echo "maximum size (256K bits) with a small instance count."
	@echo

clean:
	rm -f libgtest.a gtest-all.o tester

libgtest.a: check
	$(CXX) $(CXXFLAGS) -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -c $(GTEST_DIR)/src/gtest-all.cc
	ar -rv libgtest.a gtest-all.o
	rm gtest-all.o

kepler: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_35 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

maxwell: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_50 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

pascal: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_60 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

volta: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_70 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

turing: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_75 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

ampere: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_80 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

ada: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_89 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

blackwell: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -arch=sm_120 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

kepler-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_35 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

maxwell-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_50 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

pascal-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_60 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

volta-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_70 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

turing-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_75 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

ampere-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_80 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

ada-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_89 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

blackwell-large: libgtest.a
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $(LARGE) -I$(GTEST_DIR)/include -I../include -arch=sm_120 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "../gtest" ]; then echo "Google Test framework required, see XMP documentation"; exit 1; fi

