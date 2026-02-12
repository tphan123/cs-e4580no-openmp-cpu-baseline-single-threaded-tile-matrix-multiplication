#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <type_traits>
#include <unistd.h>

#include "mm.h"
#include "ppc.h"
#include "tests.h"

// if openmp is available, let's also speed up validation
#if _OPENMP
#define CONDITIONALLY_PARALLEL _Pragma("omp parallel for")
#else
#define CONDITIONALLY_PARALLEL
#endif

static bool verify(const input &input, const std::int32_t *result, std::int32_t *errors) {
    bool error = false;
    for (int i = 0; i < input.m; ++i) {
        for (int j = 0; j < input.n; ++j) {
            std::int64_t sum = 0;
            for (int k = 0; k < input.k; ++k) {
                sum += input.a[k + i * input.k] * input.b[j + k * input.n];
            }
            std::int64_t diff = result[j + input.n * i] - sum;
            errors[j + input.n * i] = diff == 0 ? 0 : 1;
            error = error || (diff != 0);
        }
    }
    return !error;
}

// Does 'iter' iterations of Freivald's algorithm and returns whether there is any difference
static bool verify_gvfa(const input &input, const std::int32_t *result, int iter) {
    ppc::random rng;
    std::vector<std::int32_t> x(input.n * iter);
    std::vector<std::int64_t> Bx(input.k * iter, 0);
    std::vector<std::int64_t> ABx(input.m * iter, 0);
    std::vector<std::int64_t> Cx(input.m * iter, 0);

    for (int j = 0; j < input.n; j++) {
        for (int k = 0; k < iter; k++) {
            x[j * iter + k] = rng.get_int32(-127, 127);
        }
    }

    // Bx
    CONDITIONALLY_PARALLEL
    for (int i = 0; i < input.k; i++) {
        for (int j = 0; j < input.n; j++) {
            std::int64_t left = input.b[i * input.n + j];
            for (int k = 0; k < iter; k++)
                Bx[i * iter + k] += left * x[j * iter + k];
        }
    }

    // ABx
    CONDITIONALLY_PARALLEL
    for (int j = 0; j < input.m; j++) {
        for (int i = 0; i < input.k; i++) {
            for (int k = 0; k < iter; k++)
                ABx[j * iter + k] += input.a[j * input.k + i] * Bx[i * iter + k];
        }
    }

    // Cx
    CONDITIONALLY_PARALLEL
    for (int j = 0; j < input.m; j++) {
        for (int i = 0; i < input.n; i++) {
            std::int64_t left = result[j * input.n + i];
            for (int k = 0; k < iter; k++)
                Cx[j * iter + k] += left * x[i * iter + k];
        }
    }

    // compare results
    for (int j = 0; j < input.m; j++) {
        for (int k = 0; k < iter; k++) {
            if (ABx[j * iter + k] != Cx[j * iter + k]) {
                return false;
            }
        }
    }

    return true;
}

void print_input_if_small(const input &input, std::unique_ptr<ppc::fdostream> &stream) {
    if (input.m <= 17 && input.n <= 17 && input.k <= 17 && (input.m * input.k) <= 128 && (input.k * input.n) <= 128) {
        *stream << "input_a\t";
        ppc::print_matrix(input.m, input.k, input.a.data(), stream);
        *stream << '\n';

        *stream << "input_b\t";
        ppc::print_matrix(input.k, input.n, input.b.data(), stream);
        *stream << '\n';
    }
}

void test_regular(const input &input, const std::vector<std::int32_t> &output, std::unique_ptr<ppc::fdostream> &stream) {
    bool pass = true;
    bool small = input.m * input.n <= 32 * 32 && input.n * input.m * input.k < 256 * 256;
    bool gvfa_pass = verify_gvfa(input, output.data(), 20);
    std::vector<std::int32_t> errors;
    if (small) {
        // calculate full reference result
        errors.resize(input.m * input.n);
        pass = verify(input, output.data(), errors.data());
        if (!gvfa_pass) {
            assert(pass == false);
        }
    }
    pass = pass && gvfa_pass;

    if (!pass) {
        *stream << "result\tfail\n";
        *stream << "size\t" << (small ? "small" : "large") << '\n';

        // small input?
        print_input_if_small(input, stream);

        // small output?
        if (input.m <= 17 && input.n <= 17 && (input.m * input.n) <= 128) {
            *stream << "output\t";
            ppc::print_matrix(input.m, input.n, output.data(), stream);
            *stream << '\n';

            *stream << "locations\t";
            ppc::print_matrix(input.m, input.n, errors.data(), stream);
            *stream << '\n';
        } else if (small && input.m < 60 && input.n < 60) {
            *stream << "locations\t";
            ppc::print_matrix(input.m, input.n, errors.data(), stream);
            *stream << '\n';
        }
    } else {
        *stream << "result\tpass\n";
    }
}

void test_tiled(const input &original, const std::vector<std::int32_t> &output, std::unique_ptr<ppc::fdostream> &stream) {
    int ts = original.tile_size;
    assert(original.m % ts == 0);
    assert(original.n % ts == 0);

    input small;
    small.m = original.m / ts;
    small.n = original.n / ts;
    small.k = original.k / ts;

    // create scaled-down versions
    small.a.resize(original.a.size() / ts / ts);
    small.b.resize(original.b.size() / ts / ts);
    for (int i = 0; i < original.m; i += ts) {
        for (int k = 0; k < original.k; k += ts) {
            small.a[i / ts * small.k + k / ts] = original.a[i * original.k + k];
        }
    }

    for (int k = 0; k < original.k; k += ts) {
        for (int j = 0; j < original.n; j += ts) {
            small.b[k / ts * small.n + j / ts] = original.b[k * original.n + j];
        }
    }

    // generate a small version of the output
    std::vector<std::int32_t> scaled_down_output(small.m * small.n);
    std::vector<bool> homogenous(small.m * small.n, true);
    for (int i = 0; i < original.m; i += ts) {
        for (int j = 0; j < original.n; j += ts) {
            scaled_down_output[i / ts * small.n + j / ts] = output[i * original.n + j] / ts;
            // Check that the code being tested produces same value for the whole tile
            for (int ii = 0; ii < ts; ++ii) {
                for (int jj = 0; jj < ts; ++jj) {
                    if (scaled_down_output[i / ts * small.n + j / ts] != output[(i + ii) * original.n + (j + jj)] / ts) {
                        homogenous[i / ts * small.n + j / ts] = false;
                    }
                }
            }
        }
    }

    // run regular validation at tile level
    std::vector<std::int32_t> errors;
    errors.resize(small.m * small.n);
    bool pass = verify(small, scaled_down_output.data(), errors.data());

    // insert homogeneity information into errors
    for (int i = 0; i < small.m; ++i) {
        for (int j = 0; j < small.n; ++j) {
            if (!homogenous[i * small.n + j]) {
                // sentinel value to indicate inconsistent tile
                errors[i * small.n + j] = 2;
            }
        }
    }

    if (!pass) {
        *stream << "result\tfail\n";
        *stream << "size\ttiled\n";
        *stream << "tile_size\t" << ts << "\n";

        // small input?
        print_input_if_small(small, stream);

        // small output?
        if (small.m <= 17 && small.n <= 17 && (small.m * small.n) <= 128) {
            *stream << "output\t";
            ppc::print_matrix(small.m, small.n, scaled_down_output.data(), stream);
            *stream << '\n';

            *stream << "locations\t";
            ppc::print_matrix(small.m, small.n, errors.data(), stream);
            *stream << '\n';
        }
    } else {
        *stream << "result\tpass\n";
    }
}

int main(int argc, const char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }
    std::unique_ptr<ppc::fdostream> stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    if (argc < 1 || argc > 2) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    bool test = false;
    if (argv[0] == std::string("--test")) {
        test = true;
        argc--;
        argv++;
    }

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return 2;
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    input input = generate_input(input_type, input_file);

    std::vector<std::int32_t> output(input.m * input.n);

    // ensure that `output` is initialized by non-zero numbers
    ppc::random rng;
    std::generate(begin(output), end(output), [&]() { return rng.get_int32(0, std::numeric_limits<std::int32_t>::max()); });

#if __NVCC__
    std::int8_t *a_data = nullptr;
    PPC_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&a_data), input.a.size() * sizeof(std::int8_t)),
                   "Failed to allocate input A on GPU");
    PPC_CUDA_CHECK(cudaMemcpy(a_data, input.a.data(), input.a.size() * sizeof(std::int8_t), cudaMemcpyHostToDevice),
                   "Failed to copy input A to GPU");

    std::int8_t *b_data = nullptr;
    PPC_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b_data), input.b.size() * sizeof(std::int8_t)),
                   "Failed to allocate input B on GPU");
    PPC_CUDA_CHECK(cudaMemcpy(b_data, input.b.data(), input.b.size() * sizeof(std::int8_t), cudaMemcpyHostToDevice),
                   "Failed to copy input B to GPU");

    std::int32_t *o_data = nullptr;
    PPC_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&o_data), output.size() * sizeof(std::int32_t)),
                   "Failed to allocate result on GPU");
    PPC_CUDA_CHECK(cudaMemcpy(o_data, output.data(), output.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "Failed to initilaize output on GPU");
#else
    // make a copy of our inputs, so our tests are still reliable if they get overwritten "accidentally"
    std::vector<std::int8_t> in_a = input.a;
    std::vector<std::int8_t> in_b = input.b;

    const std::int8_t *a_data = in_a.data();
    const std::int8_t *b_data = in_b.data();
    std::int32_t *o_data = output.data();
#endif

    ppc::setup_cuda_device();
    ppc::perf timer;
    timer.start();
    gemm(input.m, input.n, input.k, a_data, b_data, o_data);
#if __NVCC__
    PPC_CUDA_CHECK(cudaDeviceSynchronize(), "Error when synchronizing device");
#endif
    timer.stop();
    timer.print_to(*stream);

#if __NVCC__
    PPC_CUDA_CHECK(cudaMemcpy(output.data(), o_data, output.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                   "Failed to fetch output from GPU");
    PPC_CUDA_CHECK(cudaFree(a_data), "Failed to free A");
    PPC_CUDA_CHECK(cudaFree(b_data), "Failed to free B");
    PPC_CUDA_CHECK(cudaFree(o_data), "Failed to free C");
#endif

    ppc::reset_cuda_device();

    *stream << "m\t" << input.m << '\n'
            << "n\t" << input.n << '\n'
            << "k\t" << input.k << '\n';

    if (test) {
        if (input.tile_size != -1) {
            test_tiled(input, output, stream);
        } else {
            test_regular(input, output, stream);
        }
    } else {
        *stream << "result\tdone\n";
    }
    *stream << std::endl;
}
