#pragma once

#include "ppc.h"
#include <cstdint>
#include <string>

struct input {
    int m;
    int n;
    int k;
    std::vector<std::int8_t> a;
    std::vector<std::int8_t> b;

    // tiled versions of a and b
    int tile_size = -1;
};

static std::vector<std::int8_t> generate_identity(int m) {
    std::vector<std::int8_t> values(m * m, 0);
    for (int i = 0; i < m; ++i) {
        values[i + i * m] = 1;
    }
    return values;
}

static std::vector<std::int8_t> generate_uniform(int m, int n, int seed) {
    std::vector<std::int8_t> values;
    values.reserve(m * n);
    ppc::random rng(seed);
    for (int i = 0; i < m * n; ++i) {
        values.push_back(rng.get_int32(-127, 127));
    }
    return values;
}

static std::vector<std::int8_t> generate_ternary(int m, int n, int seed) {
    std::vector<std::int8_t> values;
    values.reserve(m * n);
    ppc::random rng(seed);
    for (int i = 0; i < m * n; ++i) {
        values.push_back(rng.get_int32(-1, 1));
    }
    return values;
}

static std::vector<std::int8_t> generate_tiled(int m, int n, int seed, int tile_size) {
    assert(m % tile_size == 0);
    assert(n % tile_size == 0);
    // first, generate small version
    std::vector<std::int8_t> small;
    small.reserve(m * n / tile_size / tile_size);
    ppc::random rng(seed);
    for (int i = 0; i < m / tile_size * n / tile_size; ++i) {
        small.push_back(rng.get_int32(-9, 9));
    }

    // now, expand to large version
    std::vector<std::int8_t> large(m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            large[i * n + j] = small[(i / tile_size) * n / tile_size + j / tile_size];
        }
    }

    return large;
}

static input generate_input(const std::string &input_type, std::ifstream &input_file) {
    ppc::random rng;
    int m, n, k;
    int tile = -1;
    CHECK_READ(input_file >> m >> n >> k);
    if (input_type == "tiled") {
        CHECK_READ(input_file >> tile);
    }
    CHECK_END(input_file);

    std::vector<std::int8_t> a;
    std::vector<std::int8_t> b;

    if (input_type == "id_x_b") {
        assert(m == k);
        a = generate_identity(m);
        b = generate_uniform(k, n, 42);
    } else if (input_type == "a_x_id") {
        assert(n == k);
        a = generate_uniform(m, k, 42);
        b = generate_identity(n);
    } else if (input_type == "uniform") {
        a = generate_uniform(m, k, 42);
        b = generate_uniform(k, n, 21);
    } else if (input_type == "ternary") {
        a = generate_ternary(m, k, 42);
        b = generate_ternary(k, n, 21);
    } else if (input_type == "tiled") {
        a = generate_tiled(m, k, 42, tile);
        b = generate_tiled(k, n, 21, tile);
    } else {
        std::cerr << "unknown input_type\n";
        std::exit(3);
    }

    return {m, n, k, a, b, tile};
}
