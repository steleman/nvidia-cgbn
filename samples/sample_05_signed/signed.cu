/***

Copyright (c) 2024-2026, Stefan Teleman.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"

/************************************************************************************************
 *  This example demonstrates CGBN's *signed* (two's complement) integer
 * support.  A signed CGBN uses exactly the same fixed-width, little-endian limb
 * encoding as an unsigned CGBN -- the most significant bit (bit BITS-1) is
 * simply reinterpreted as a sign bit.  As a result, load/store, addition,
 * subtraction, negation and the low-half multiply need no signed variant; only
 * the operations whose result depends on the sign get a cgbn_signed_* wrapper
 * (see cgbn/cgbn_signed.h).
 *
 *  For each instance the GPU kernel computes, on two BITS-bit signed operands a
 * and b:
 *
 *      cmp     = signed compare(a, b)                 (-1, 0, or 1)
 *      q, r    = signed truncated division a / b      (q toward zero, r has the
 * sign of a) shifted = arithmetic right shift a >> SHIFT     (sign-extending)
 *      mul_lo  = low  BITS bits of the signed product a * b
 *      mul_hi  = high BITS bits of the signed product a * b
 *
 *  Every result is then verified on the CPU with GMP, interpreting the limbs as
 * two's complement.
 ************************************************************************************************/

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 8
#define BITS 1024
#define SHIFT 173
#define INSTANCES 50000

typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> q;       // trunc(a / b)
  cgbn_mem_t<BITS> r;       // a - q*b
  cgbn_mem_t<BITS> shifted; // a >> SHIFT, arithmetic
  cgbn_mem_t<BITS> mul_lo;  // low  BITS bits of a*b
  cgbn_mem_t<BITS> mul_hi;  // high BITS bits of a*b
  int32_t cmp;              // signed compare(a, b)
  int32_t pad[31];          // keep the struct 128-byte aligned
} instance_t;

// import little-endian limbs and reinterpret as a two's complement signed value
// of width 32*count
void from_signed_mem(mpz_t x, const uint32_t* limbs, uint32_t count) {
  mpz_import(x, count, -1, sizeof(uint32_t), 0, 0, limbs);
  if ((limbs[count - 1] >> 31) != 0) {
    mpz_t modulus;
    mpz_init(modulus);
    mpz_ui_pow_ui(modulus, 2, 32 * count);
    mpz_sub(x, x, modulus);
    mpz_clear(modulus);
  }
}

// write x as a two's complement value of width 32*count into little-endian
// limbs (zero filled)
void to_signed_mem(uint32_t* limbs, uint32_t count, const mpz_t x) {
  mpz_t modulus, reduced;
  size_t words = 0;

  mpz_init(modulus);
  mpz_init(reduced);
  mpz_ui_pow_ui(modulus, 2, 32 * count);
  mpz_mod(reduced, x, modulus); // mpz_mod yields a value in [0, modulus): the
                                // two's complement bits
  for (uint32_t index = 0; index < count; index++)
    limbs[index] = 0;
  mpz_export(limbs, &words, -1, sizeof(uint32_t), 0, 0, reduced);
  mpz_clear(modulus);
  mpz_clear(reduced);
}

instance_t* generate_instances(uint32_t count) {
  instance_t* instances = (instance_t*)malloc(sizeof(instance_t) * count);

  for (int index = 0; index < count; index++) {
    random_words(instances[index].a._limbs, BITS / 32);
    random_words(instances[index].b._limbs, BITS / 32);
    // make sure the divisor is non-zero (astronomically unlikely to be zero,
    // but be safe)
    instances[index].b._limbs[0] |= 1;
  }
  return instances;
}

void verify_results(instance_t* instances, uint32_t count) {
  mpz_t a, b, q, r, t, expected;
  uint32_t words[BITS / 32];

  mpz_init(a);
  mpz_init(b);
  mpz_init(q);
  mpz_init(r);
  mpz_init(t);
  mpz_init(expected);

  for (int index = 0; index < count; index++) {
    from_signed_mem(a, instances[index].a._limbs, BITS / 32);
    from_signed_mem(b, instances[index].b._limbs, BITS / 32);

    // signed compare
    int cmp = mpz_cmp(a, b);
    cmp = (cmp > 0) ? 1 : (cmp < 0) ? -1 : 0;
    if (cmp != instances[index].cmp) {
      printf("signed compare failed on instance %d (cpu %d, gpu %d)\n", index,
             cmp, instances[index].cmp);
      return;
    }

    // truncated division / remainder
    mpz_tdiv_qr(q, r, a, b);
    to_signed_mem(words, BITS / 32, q);
    if (compare_words(words, instances[index].q._limbs, BITS / 32) != 0) {
      printf("signed div failed on instance %d\n", index);
      return;
    }
    to_signed_mem(words, BITS / 32, r);
    if (compare_words(words, instances[index].r._limbs, BITS / 32) != 0) {
      printf("signed rem failed on instance %d\n", index);
      return;
    }

    // arithmetic right shift == floor(a / 2^SHIFT)
    mpz_fdiv_q_2exp(t, a, SHIFT);
    to_signed_mem(words, BITS / 32, t);
    if (compare_words(words, instances[index].shifted._limbs, BITS / 32) != 0) {
      printf("signed shift_right failed on instance %d\n", index);
      return;
    }

    // full signed product, split into low and high BITS-bit halves
    mpz_mul(t, a, b);
    to_signed_mem(
        words, BITS / 32,
        t); // low  BITS bits (two's complement of the product mod 2^BITS)
    if (compare_words(words, instances[index].mul_lo._limbs, BITS / 32) != 0) {
      printf("signed mul_wide (low) failed on instance %d\n", index);
      return;
    }
    mpz_fdiv_q_2exp(
        expected, t,
        BITS); // arithmetic (sign-extending) high half of the 2*BITS product
    to_signed_mem(words, BITS / 32, expected);
    if (compare_words(words, instances[index].mul_hi._limbs, BITS / 32) != 0) {
      printf("signed mul_wide (high) failed on instance %d\n", index);
      return;
    }
  }

  mpz_clear(a);
  mpz_clear(b);
  mpz_clear(q);
  mpz_clear(r);
  mpz_clear(t);
  mpz_clear(expected);
  printf("All results match\n");
}

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__global__ void kernel_signed(cgbn_error_report_t* report,
                              instance_t* instances, uint32_t count) {
  int32_t instance;

  instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= count)
    return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t bn_env(bn_context.env<env_t>());
  env_t::cgbn_t a, b, q, r, s;
  env_t::cgbn_wide_t w;

  cgbn_load(bn_env, a, &(instances[instance].a));
  cgbn_load(bn_env, b, &(instances[instance].b));

  instances[instance].cmp = cgbn_signed_compare(bn_env, a, b);

  cgbn_signed_div_rem(bn_env, q, r, a, b);
  cgbn_signed_shift_right(bn_env, s, a, SHIFT);
  cgbn_signed_mul_wide(bn_env, w, a, b);

  cgbn_store(bn_env, &(instances[instance].q), q);
  cgbn_store(bn_env, &(instances[instance].r), r);
  cgbn_store(bn_env, &(instances[instance].shifted), s);
  cgbn_store(bn_env, &(instances[instance].mul_lo), w._low);
  cgbn_store(bn_env, &(instances[instance].mul_hi), w._high);
}

int main() {
  instance_t *instances, *gpuInstances;
  cgbn_error_report_t* report;

  printf("Generating instances ...\n");
  instances = generate_instances(INSTANCES);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void**)&gpuInstances, sizeof(instance_t) * INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t) * INSTANCES,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");
  // 8 threads per instance, 128 threads (16 instances) per block
  kernel_signed<<<(INSTANCES + 15) / 16, 128>>>(report, gpuInstances,
                                                INSTANCES);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t) * INSTANCES,
                        cudaMemcpyDeviceToHost));

  printf("Verifying the results ...\n");
  verify_results(instances, INSTANCES);

  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}
