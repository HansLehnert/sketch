#pragma once

#include <cstring>

template <typename T, int n_hash, int n_bits>
using Sketch = T[n_hash][1 << n_bits];

template <typename T, int n_hash, int n_bits>
class SketchSet {
    Sketch<T, n_hash, n_bits>* sketches;

public:
    SketchSet(int n_sketch) {
        sketches = new Sketch<T, n_hash, n_bits>[n_sketch];
        memset(sketches, 0, sizeof(Sketch<T, n_hash, n_bits>) * n_sketch);
    }

    ~SketchSet() {
        delete[] sketches;
    }

    Sketch<T, n_hash, n_bits>& operator[](int n) {
        return sketches[n];
    }
};
