// Fixed size arraay that packs sub-byte sized data

#include <cstring>
#include <functional>

template <int symbol_size, int length, typename BaseType = unsigned long>
struct PackedArray {
private:
    static_assert(sizeof(BaseType) * 8 >= symbol_size);

    constexpr static int values_per_data = sizeof(BaseType) * 8 / symbol_size;
    constexpr static int total_data =
        (length + values_per_data - 1) / values_per_data;;

public:
    BaseType data[total_data];

    PackedArray() {
        memset(data, 0, sizeof(data));
    }

    void set(int index, unsigned long value) {
        data[index / values_per_data] |=
            value << (index % values_per_data * symbol_size);
    }

    unsigned long get(int index) const {
        unsigned long result =
            (data[index / values_per_data]
                >> (index % values_per_data * symbol_size))
            & ~(~0UL << symbol_size);

        return result;
    }

    bool operator==(const PackedArray<symbol_size, length>& other) const {
        for (int i = 0; i < total_data; i++) {
            if (data[i] != other.data[i])
                return false;
        }
        return true;
    };

    friend std::hash<PackedArray<symbol_size, length, BaseType>>;
};


namespace std {
    template<int symbol_size, int length, typename BaseType>
    struct hash<PackedArray<symbol_size, length, BaseType>> {
        size_t operator()(
                const PackedArray<symbol_size, length>& value
        ) const noexcept {
            size_t result = 0;
            for (int i = 0; i < value.total_data; i++)
                result ^= value.data[i];
            return result;
        }
    };
}
