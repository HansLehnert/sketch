G++ = g++
NVCC = nvcc
CPP_FLAGS = -std=c++11 -O3 -pthread
AVX_FLAGS = -mavx -mavx2
CU_FLAGS = -std=c++11 -O3
MKDIR = mkdir -p

USE_CUDA := $(shell command -v nvcc 2> /dev/null)
USE_AVX := $(shell grep avx2 /proc/cpuinfo)


EXECUTABLES += sketch
# EXECUTABLES += sketch_multithread

ifdef USE_AVX
	# EXECUTABLES += sketch_avx
	# EXECUTABLES += sketch_avx_multithread
	# EXECUTABLES += sketch_avx_multithread_approx
	EXECUTABLES += sketch_avx_pipelined
endif

ifdef USE_CUDA
	EXECUTABLES += sketch_cu
	# EXECUTABLES += sketch_cu_pipelined
	# EXECUTABLES += sketch_cu_approx
endif

DATASET = \
	./data/test.fasta \
	./data/control.fasta \
	10 20 365 308 257 161 150 145 145 145 145 145 145

all: $(addprefix bin/, $(EXECUTABLES))

run: all
	@$(MKDIR) out/
	@$(foreach exec, $(EXECUTABLES), echo $(exec);./bin/$(exec) $(DATASET) > out/$(exec).txt;)

bin/%.o: src/%.cpp
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) -c $^ -o $@

bin/sketch: src/sketch.cpp bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_multithread: src/sketch_multithread.cpp bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_avx: src/sketch_avx.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_multithread: src/sketch_avx_multithread.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_multithread_approx: src/sketch_avx_multithread_approx.cpp bin/fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_avx_pipelined: src/sketch_avx_pipelined.cpp bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_cu: src/sketch.cu bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

bin/sketch_cu_approx: src/sketch_approx.cu bin/fasta.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

bin/sketch_cu_pipelined: src/sketch_pipelined.cu bin/fasta.o bin/MappedFile.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CU_FLAGS) --compiler-options="$(AVX_FLAGS) $(CPP_FLAGS)" $^ -o $@

clean:
	rm -r bin/
