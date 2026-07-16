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
 *  This example demonstrates CGBN division and square root at a *very large*
 * size: 256K bits (262144 bits), the top of the supported range.
 *
 *  Division, remainder and square root reduce the leading word of the operand
 * using an internal reciprocal.  When BITS/32/TPI (the limbs-per-thread,
 * "LIMBS") is <= TPI, that reciprocal is computed with a warp-distributed
 * Newton-Raphson.  At 256K bits with TPI=32, LIMBS=256 > TPI, so the library
 * takes the "multi" reciprocal path (dlimbs_algs_multi), which is what makes
 *  sizes above 32K bits work at all.  This sample exercises exactly that path.
 *
 *  For each instance the GPU kernel computes, on two BITS-bit unsigned operands
 * a and b (b != 0):
 *
 *      q, r = a / b, a % b      (unsigned truncated division)
 *      s    = isqrt(a)          (floor of the integer square root)
 *
 *  Every result is verified on the CPU with GMP:  q*b + r == a with 0 <= r < b,
 * and s*s <= a < (s+1)*(s+1).
 *
 *  Note: the multi reciprocal is serial and redundant across the warp, so
 * 256K-bit division is much slower per instance than at 32K bits -- a few
 * thousand instances is plenty for a demo. Prefer the largest TPI (32) at large
 * sizes to keep LIMBS, and hence the serial window, small.
 ************************************************************************************************/

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS                                                                   \
  262144 // 256K bits: LIMBS = 262144/32/32 = 256 > TPI -> multi reciprocal path
#define INSTANCES 2000

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> q; // a / b
  cgbn_mem_t<BITS> r; // a % b
  cgbn_mem_t<BITS> s; // isqrt(a)
} instance_t;

// support routine to generate random instances
instance_t* generate_instances(uint32_t count) {
  instance_t* instances = (instance_t*)malloc(sizeof(instance_t) * count);

  for (int index = 0; index < count; index++) {
    random_words(instances[index].a._limbs, BITS / 32);
    random_words(instances[index].b._limbs, BITS / 32);
    // vary the divisor magnitude so the quotient spans many words on some
    // instances
    if ((index & 1) == 0)
      for (uint32_t word = BITS / 32 / 2; word < BITS / 32; word++)
        instances[index].b._limbs[word] = 0;
    instances[index].b._limbs[0] |= 1; // guarantee b is non-zero
  }
  return instances;
}

// support routine to verify the GPU results using GMP on the CPU
void verify_results(instance_t* instances, uint32_t count) {
  mpz_t a, b, q, r, s, eq, er, es, s2, s2p;

  mpz_init(a);
  mpz_init(b);
  mpz_init(q);
  mpz_init(r);
  mpz_init(s);
  mpz_init(eq);
  mpz_init(er);
  mpz_init(es);
  mpz_init(s2);
  mpz_init(s2p);

  for (int index = 0; index < count; index++) {
    to_mpz(a, instances[index].a._limbs, BITS / 32);
    to_mpz(b, instances[index].b._limbs, BITS / 32);
    to_mpz(q, instances[index].q._limbs, BITS / 32);
    to_mpz(r, instances[index].r._limbs, BITS / 32);
    to_mpz(s, instances[index].s._limbs, BITS / 32);

    // expected quotient / remainder (truncated, which for unsigned == floor)
    mpz_tdiv_qr(eq, er, a, b);
    if (mpz_cmp(q, eq) != 0) {
      printf("gpu div kernel failed on instance %d (quotient)\n", index);
      return;
    }
    if (mpz_cmp(r, er) != 0) {
      printf("gpu div kernel failed on instance %d (remainder)\n", index);
      return;
    }

    // expected integer square root, and the bracketing check s^2 <= a < (s+1)^2
    mpz_sqrt(es, a);
    if (mpz_cmp(s, es) != 0) {
      printf("gpu sqrt kernel failed on instance %d\n", index);
      return;
    }
    mpz_mul(s2, s, s);
    mpz_add_ui(s2p, s, 1);
    mpz_mul(s2p, s2p, s2p);
    if (mpz_cmp(s2, a) > 0 || mpz_cmp(a, s2p) >= 0) {
      printf("sqrt bracket check failed on instance %d\n", index);
      return;
    }
  }
  printf("All results match\n");

  mpz_clear(a);
  mpz_clear(b);
  mpz_clear(q);
  mpz_clear(r);
  mpz_clear(s);
  mpz_clear(eq);
  mpz_clear(er);
  mpz_clear(es);
  mpz_clear(s2);
  mpz_clear(s2p);
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_large_div(cgbn_error_report_t* report,
                                 instance_t* instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (instance >= count)
    return;

  context_t bn_context(cgbn_report_monitor, report,
                       instance); // construct a context
  env_t bn_env(
      bn_context.env<env_t>()); // construct an environment for 256K-bit math
  env_t::cgbn_t a, b, q, r, s;  // 256K-bit values (spread across a warp)

  cgbn_load(bn_env, a, &(instances[instance].a)); // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b)); // load my instance's b value

  cgbn_div_rem(bn_env, q, r, a, b); // q=a/b, r=a%b
  cgbn_sqrt(bn_env, s, a);          // s=isqrt(a)

  cgbn_store(bn_env, &(instances[instance].q), q); // store quotient
  cgbn_store(bn_env, &(instances[instance].r), r); // store remainder
  cgbn_store(bn_env, &(instances[instance].s), s); // store square root
}

int main() {
  instance_t *instances, *gpuInstances;
  cgbn_error_report_t* report;

  printf("Generating %d instances of %d-bit division and square root ...\n",
         INSTANCES, BITS);
  instances = generate_instances(INSTANCES);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void**)&gpuInstances, sizeof(instance_t) * INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t) * INSTANCES,
                        cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel (256K-bit div_rem + sqrt) ...\n");
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_large_div<<<(INSTANCES + 3) / 4, 128>>>(report, gpuInstances,
                                                 INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and
  // check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t) * INSTANCES,
                        cudaMemcpyDeviceToHost));

  printf("Verifying the results ...\n");
  verify_results(instances, INSTANCES);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}
