G++ = g++
NVCC = nvcc
CPP_FLAGS = -std=c++11 -O3 -pthread
AVX_FLAGS = -mavx -mavx2
CU_FLAGS = -std=c++11 -O3
MKDIR = mkdir -p

USE_CUDA := $(shell command -v nvcc 2> /dev/null)
USE_AVX := $(shell grep avx2 /proc/cpuinfo)


EXECUTABLES = sketch sketch_mmap sketch_multithread

ifdef USE_AVX
	EXECUTABLES += sketch_avx sketch_avx_multithread sketch_avx_multithread_approx
endif

ifdef USE_CUDA
	EXECUTABLES += sketch_cu sketch_cu_approx sketch_cu_pipelined
endif

DATASET = ./data/test.fasta ./data/control.fasta

all: $(addprefix bin/, $(EXECUTABLES))

run: all
	./bin/sketch_mmap $(DATASET)
	./bin/sketch_multithread $(DATASET)
ifdef USE_AVX
# ./bin/sketch_avx $(DATASET)
# ./bin/sketch_avx_multithread $(DATASET)
# ./bin/sketch_avx_multithread_approx $(DATASET)
endif
ifdef USE_CUDA
	./bin/sketch_cu $(DATASET)
	./bin/sketch_cu_pipelined $(DATASET)
# ./bin/sketch_cu_approx $(DATASET)
endif

bin/%.o: %.cpp
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) -c $^ -o $@

bin/sketch: sketch.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_mmap: sketch_mmap.cpp bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_multithread: sketch_multithread.cpp bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_avx: sketch_avx.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_multithread: sketch_avx_multithread.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_multithread_approx: sketch_avx_multithread_approx.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_cu: sketch.cu bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

bin/sketch_cu_approx: sketch_approx.cu bin/fasta.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

bin/sketch_cu_pipelined: sketch_pipelined.cu bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

clean:
	rm -r bin/
