G++ = g++
NVCC = nvcc
CPP_FLAGS = -std=c++11 -O3
AVX_FLAGS = -mavx -mavx2
CU_FLAGS = -std=c++11 -O3
MKDIR = mkdir -p

USE_CUDA := $(shell command -v nvcc 2> /dev/null)
USE_AVX := $(shell grep avx2 /proc/cpuinfo)


EXECUTABLES = sketch


ifdef USE_AVX
	EXECUTABLES += sketch_avx sketch_avx_multithread
endif

ifdef USE_CUDA
	EXECUTABLES += sketch_cu
endif

DATASET = ./data/test.fasta

all: $(addprefix bin/, $(EXECUTABLES))

run: all
	./bin/sketch $(DATASET)
ifdef USE_AVX
	./bin/sketch_avx $(DATASET)
	./bin/sketch_avx_multithread $(DATASET)
endif
ifdef USE_CUDA
	./bin/sketch_cu $(DATASET)
endif

bin/%.o: %.cpp
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) -c $^ -o $@

bin/sketch: sketch.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_avx: sketch_avx.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_multithread: sketch_avx_multithread.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) -pthread $^ -o $@

bin/sketch_cu: sketch.cu bin/fasta.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

clean:
	rm -r bin/
