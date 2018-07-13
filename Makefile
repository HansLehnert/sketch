G++ = g++
FLAGS = -std=c++11 -O3 -mavx -mavx2

all: sketch sketch_avx

run: all
	./sketch

sketch: sketch.cpp
	$(G++) $(FLAGS) $^ -o $@

sketch_avx: sketch_avx.cpp
	$(G++) $(FLAGS) $^ -o $@

clean:
	rm sketch
