#pragma once
#include <cstdint>

void gemm(int m, int n, int k, const std::int8_t *A, const std::int8_t *B, std::int32_t *C);
