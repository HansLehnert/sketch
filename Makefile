G++ = g++
FLAGS = -std=c++11 -O3 -mavx -mavx2

all: sketch

run: all
	./sketch

fasta.o: fasta.cpp
	$(G++) $(FLAGS) -c $^ -o $@

sketch: sketch.cpp fasta.o
	$(G++) $(FLAGS) $^ -o $@

clean:
	rm sketch
