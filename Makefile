G++ = g++
FLAGS = -std=c++11 -O3 -mavx -mavx2
MKDIR = mkdir -p

all: bin/sketch bin/sketch_avx

run: all
	./bin/sketch
	./bin/sketch_avx

fasta.o: fasta.cpp
	$(G++) $(FLAGS) -c $^ -o $@

bin/sketch: sketch.cpp fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(FLAGS) $^ -o $@

bin/sketch_avx: sketch_avx.cpp fasta.o
	@$(MKDIR) $(@D)
	$(G++) $(FLAGS) $^ -o $@

clean:
	rm -r bin/
