G++ = g++
NVCC = nvcc
CPP_FLAGS = -std=c++11 -O3
AVX_FLAGS = -mavx -mavx2
MKDIR = mkdir -p

all: bin/sketch bin/sketch_avx bin/sketch_cu

run: all
	./bin/sketch
	./bin/sketch_avx
	./bin/sketch_cu

fasta.o: fasta.cpp
	$(G++) $(CPP_FLAGS) -c $^ -o $@

bin/sketch: sketch.cpp fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $^ -o $@

bin/sketch_avx: sketch_avx.cpp fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(CPP_FLAGS) $(AVX_FLAGS) $^ -o $@

bin/sketch_cu: sketch.cu fasta.o
	@$(MKDIR) $(@D)
	$(NVCC) $(CPP_FLAGS) $^ -o $@

clean:
	rm -r bin/
