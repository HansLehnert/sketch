// Simple fixed size hash table
#pragma once

#include <vector>

template <int bits>
struct HashTable {
    static constexpr uint64_t mask = ~(~0 << bits);
    static constexpr uint64_t n_slots = 1 << bits;
    // static constexpr size = n_slots * sizeof(Slot);

    struct Slot {
        bool used;
        uint64_t key;
        int32_t value;
    } slots[n_slots];
};

template <int bits>
__device__ void hashTableInsert(
        HashTable<bits>* table,
        uint64_t key,
        int32_t value
) {
    uint64_t hash = key & table->mask;  // The last bits are used as fast hash

    while (table->slots[hash].used && table->slots[hash].key != key) {
        hash = (hash + 1) & table->mask;
    }

    table->slots[hash].used = true;
    table->slots[hash].key = key;
    table->slots[hash].value = value;
}

template <int bits>
__device__ bool hashTableGet(
        HashTable<bits>* table,
        uint64_t key,
        int32_t** value
) {
    uint64_t hash = key & table->mask;  // The last bits are used as fast hash

    while (table->slots[hash].key != key && table->slots[hash].used) {
        hash = (hash + 1) & table->mask;
    }

    *value = &(table->slots[hash].value);
    return table->slots[hash].used;
}
