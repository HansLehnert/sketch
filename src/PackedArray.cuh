// Fixed size arraay that packs sub-byte sized data
#pragma once

#include <cstring>
#include <functional>


template <int symbol_size, typename BaseType = uint64_t>
__device__ void packedArraySet(
        uint64_t* array,
        uint64_t index,
        uint64_t value
) {
    *array |= value << (index * symbol_size);
}

template <int symbol_size, typename BaseType = uint64_t>
__device__ uint64_t packedArrayGet(uint64_t* array, uint64_t index) {
    return (*array >> (index * symbol_size)) & ~(~0UL << symbol_size);
}
