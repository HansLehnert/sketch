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
        uint32_t key;
        uint64_t value;
    } slots[n_slots];
};

template <int bits>
__device__ void hashTableInsert(
        HashTable<bits>* table,
        uint32_t key,
        uint64_t value
) {
    uint32_t hash = key & table->mask;  // The last bits are used as fast hash

    while (table->slots[hash].used && table->slots[hash].key != key) {
        hash = (hash + 1) & table->mask;
    }

    table->slots[hash].used = true;
    table->slots[hash].key = key;
    table->slots[hash].value = value;
}
