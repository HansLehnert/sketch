G++ = g++
NVCC = nvcc
CPP_FLAGS = -std=c++11 -O3 -pthread
AVX_FLAGS = -mavx -mavx2
CU_FLAGS = -std=c++11 -O3
MKDIR = mkdir -p

USE_CUDA := $(shell command -v nvcc 2> /dev/null)
USE_AVX := $(shell grep avx2 /proc/cpuinfo)


# EXECUTABLES += sketch
EXECUTABLES += sketch_mmap
EXECUTABLES += sketch_multithread

ifdef USE_AVX
	# EXECUTABLES += sketch_avx
	# EXECUTABLES += sketch_avx_multithread
	# EXECUTABLES += sketch_avx_multithread_approx
	EXECUTABLES += sketch_avx_pipelined
endif

ifdef USE_CUDA
	EXECUTABLES += sketch_cu
	EXECUTABLES += sketch_cu_pipelined
	# EXECUTABLES += sketch_cu_approx
endif

DATASET = ./data/test.fasta ./data/control.fasta

all: $(addprefix bin/, $(EXECUTABLES))

run: all
	@$(MKDIR) out/
# 	./bin/sketch_mmap $(DATASET) > out/mmap.txt
# 	./bin/sketch_multithread $(DATASET) > out/multithread.txt
# ifdef USE_AVX
# # ./bin/sketch_avx $(DATASET) > out/avx.txt
# # ./bin/sketch_avx_multithread $(DATASET) > out/avx_mutithread.tx
# # ./bin/sketch_avx_multithread_approx $(DATASET) > out/avx_multithread_approx.txt
# endif
# ifdef USE_CUDA
# 	./bin/sketch_cu $(DATASET) > out/cuda.txt
# 	./bin/sketch_cu_pipelined $(DATASET) > out/cuda_pipelined.txt
# # ./bin/sketch_cu_approx $(DATASET) > out/cuda_approx.txt
# endif
	@$(foreach exec, $(EXECUTABLES), echo $(exec);./bin/$(exec) $(DATASET) > out/$(exec).txt;)

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

bin/sketch_avx_pipelined: sketch_avx_pipelined.cpp bin/fasta.o bin/MappedFile.o
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
