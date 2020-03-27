#pragma once

#include <immintrin.h>
#include <limits>


union vec128 {
    __m128i v;
    uint16_t s[8];
    uint32_t i[4];
    uint64_t l[2];
};

union vec256 {
    __m256i v;
    __m128i v128[2];
    uint16_t s[16];
    uint32_t i[8];
    uint64_t l[4];
    struct {
        vec128 lo;
        vec128 hi;
    };
};


constexpr unsigned int ceilToPowerOf2(unsigned int val) {
    unsigned int result = val - 1;
    result |= result >> 1;
    result |= result >> 2;
    result |= result >> 4;
    result |= result >> 8;
    result |= result >> 16;
    return result + 1;
}


constexpr vec256 generateMask(unsigned int n) {
    vec256 result = {0};

    unsigned int m = ceilToPowerOf2(n);
    for (unsigned int i = 0; i < 8; i++)
        result.i[i] = i % m < n ? 0 : std::numeric_limits<int32_t>::max();

    return result;
}
