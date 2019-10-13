#include <cstring>
#include <functional>

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

template <int symbol_size, int length, typename BaseType = unsigned long>
struct PackedArray {
private:
    constexpr static int total_bits = symbol_size * length;
    constexpr static int total_bytes = (total_bits + 7) / 8;
    constexpr static int total_data =
        (total_bytes + sizeof(BaseType) - 1) / sizeof(BaseType);

public:
    BaseType data;

    PackedArray() : data(0) {}

    void set(int index, unsigned int value) {
        data |= value << (index * symbol_size);
    }

    unsigned int get(int index) const {
        return (data >> (index * symbol_size)) & ~(~0UL << symbol_size);
    }

    void clear() {
        data = 0;
    }

    bool operator==(const PackedArray<symbol_size, length>& other) const {
        return data == other.data;
    }

    /*unsigned long data[total_data] = {0};

    void set(int index, unsigned int value) {
        data[index * symbol_size / sizeof(unsigned long)] |=
            value << ((index * symbol_size) % sizeof(unsigned long));
    }

    unsigned int get(int index) {
        unsigned int value = (data[index * symbol_size / sizeof(unsigned long)]
            >> ((index * symbol_size) % sizeof(unsigned long))) &
            ~(~0UL << symbol_size);

        return value;
    }*/

    /*bool operator==(PackedArray<symbol_size, length>& other) const {
        for (int i = 0; i < total_data; i++) {
            if (this.data[i] != other.data[i])
                return false;
        }
        return true;
    }*/;
};

namespace std {
    template<>
    template<int symbol_size, int length, typename BaseType>
    struct hash<PackedArray<symbol_size, length, BaseType>> {
        size_t operator()(
            const PackedArray<symbol_size, length>& key) const noexcept
        {
            return key.data;
        }
    };
}
