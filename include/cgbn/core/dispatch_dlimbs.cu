/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

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

namespace cgbn {

typedef enum {
  dlimbs_algs_common,
  dlimbs_algs_half,
  dlimbs_algs_full,
  dlimbs_algs_multi
} dlimbs_algs_t;

template <class core, dlimbs_algs_t implementation> class dispatch_dlimbs_t;

template <class core> class dispatch_dlimbs_t<core, dlimbs_algs_common> {
public:
  static const uint32_t TPI = core::TPI;
  static const uint32_t LIMBS = core::LIMBS;
  static const uint32_t DLIMBS = core::DLIMBS;
  static const uint32_t LIMB_OFFSET = DLIMBS * TPI - LIMBS;

  __device__ __forceinline__ static void
  dlimbs_scatter(uint32_t r[DLIMBS], const uint32_t x[LIMBS],
                 const uint32_t source_thread) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t;

    mpzero<DLIMBS>(r);
#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, x[index], source_thread, TPI);
      r[(index + LIMB_OFFSET) % DLIMBS] =
          (group_thread == (index + LIMB_OFFSET) / DLIMBS)
              ? t
              : r[(index + LIMB_OFFSET) % DLIMBS];
    }
  }

  __device__ __forceinline__ static void
  dlimbs_gather(uint32_t r[LIMBS], const uint32_t x[DLIMBS],
                const uint32_t destination_thread) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t;

#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, x[(index + LIMB_OFFSET) % DLIMBS],
                      (index + LIMB_OFFSET) / DLIMBS, TPI);
      r[index] = (group_thread == destination_thread) ? t : r[index];
    }
  }

  __device__ __forceinline__ static void
  dlimbs_all_gather(uint32_t r[LIMBS], const uint32_t x[DLIMBS]) {
    uint32_t sync = core::sync_mask();

#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++)
      r[index] = __shfl_sync(sync, x[(index + LIMB_OFFSET) % DLIMBS],
                             (index + LIMB_OFFSET) / DLIMBS, TPI);
  }
};

template <class core> class dispatch_dlimbs_t<core, dlimbs_algs_half> {
public:
  static const uint32_t TPI = core::TPI;
  static const uint32_t LIMBS = core::LIMBS;
  static const uint32_t DLIMBS = core::DLIMBS;
  static const uint32_t LIMB_OFFSET = DLIMBS * TPI - LIMBS;

  // these algorithms require that LIMBS<=TPI/2

  __device__ __forceinline__ static void
  dlimbs_approximate(uint32_t approx[DLIMBS], const uint32_t denom[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t x, d0, d1, x0, x1, x2, est, a, h, l;
    int32_t c, top;

    // computes (beta^2 - 1) / denom - beta, where beta=1<<32*LIMBS

    x = 0xFFFFFFFF - denom[0];

    d1 = __shfl_sync(sync, denom[0], TPI - 1, TPI);
    d0 = __shfl_sync(sync, denom[0], TPI - 2, TPI);

    approx[0] = 0;
    a = uapprox(d1);

#pragma nounroll
    for (int32_t thread = LIMBS - 1; thread >= 0; thread--) {
      x0 = __shfl_sync(sync, x, TPI - 3, TPI);
      x1 = __shfl_sync(sync, x, TPI - 2, TPI);
      x2 = __shfl_sync(sync, x, TPI - 1, TPI);
      est = udiv(x0, x1, x2, d0, d1, a);

      l = madlo_cc(est, denom[0], 0);
      h = madhic(est, denom[0], 0);

      x = sub_cc(x, h);
      c = subc(0, 0); // thread TPI-1 is zero

      top = __shfl_sync(sync, x, TPI - 1, TPI);
      x = __shfl_sync(sync, x, threadIdx.x - 1, TPI);
      c = __shfl_sync(sync, c, threadIdx.x - 1, TPI);

      x = sub_cc(x, l);
      c = subc(c, 0);

      if (top + core::resolve_sub(c, x) < 0) {
        // means a correction is required, should be very rare
        x = add_cc(x, denom[0]);
        c = addc(0, 0);
        core::fast_propagate_add(c, x);
        est--;
      }
      approx[0] = (group_thread == thread + TPI - LIMBS) ? est : approx[0];
    }
  }

  __device__ __forceinline__ static uint32_t
  dlimbs_sqrt_rem_wide(uint32_t s[DLIMBS], uint32_t r[DLIMBS],
                       const uint32_t lo[DLIMBS], const uint32_t hi[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t x, x0, x1, t0, t1, divisor, approx, p, q, c;

    // computes s=sqrt(x), r=x-s^2, where x=(hi<<32*LIMBS) + lo

    t0 = __shfl_sync(sync, lo[0], threadIdx.x + LIMBS, TPI);
    x = hi[0] | t0;
    x0 = __shfl_sync(sync, x, TPI - 2, TPI);
    x1 = __shfl_sync(sync, x, TPI - 1, TPI);

    divisor = usqrt(x0, x1);
    approx = uapprox(divisor);

    t0 = madlo(divisor, divisor, 0);
    t1 = madhi(divisor, divisor, 0);
    x0 = sub_cc(x0, t0);
    x1 = subc(x1, t1);

    x = __shfl_up_sync(sync, x, 1, TPI);
    x = (group_thread == TPI - 1) ? x0 : x;
    s[0] = (group_thread == TPI - 1) ? divisor + divisor
                                     : 0; // silent 1 at the top of s

#pragma nounroll
    for (int32_t index = TPI - 2; index >= (int32_t)(TPI - LIMBS); index--) {
      x0 = __shfl_sync(sync, x, TPI - 1, TPI);
      q = usqrt_div(x0, x1, divisor, approx);
      s[0] = (group_thread == index) ? q : s[0];

      p = madhi(q, s[0], 0);
      x = sub_cc(x, p);
      c = subc(0, 0);
      core::fast_propagate_sub(c, x);

      x1 = __shfl_sync(sync, x, TPI - 1, TPI) -
           q; // we subtract q because of the silent 1 at the top of s
      x = __shfl_up_sync(sync, x, 1, TPI);
      p = madlo(q, s[0], 0);
      x = sub_cc(x, p);
      c = subc(0, 0);
      x1 -= core::fast_propagate_sub(c, x);

      while (0 > (int32_t)x1) {
        x1++;
        q--;

        // correction step: add q and s
        x = add_cc(x, (group_thread == index) ? q : 0);
        c = addc(0, 0);
        x = add_cc(x, s[0]);
        c = addc(c, 0);

        x1 += core::resolve_add(c, x);

        // update s
        s[0] = (group_thread == index) ? q : s[0];
      }
      s[0] = (group_thread == index + 1) ? s[0] + (q >> 31) : s[0];
      s[0] = (group_thread == index) ? q + q : s[0];
    }
    t0 = __shfl_down_sync(sync, s[0], 1, TPI);
    t0 = (group_thread == TPI - 1) ? 1 : t0;
    s[0] = uright_wrap(s[0], t0, 1);
    r[0] = x;
    return x1;
  }

  __device__ __forceinline__ static void
  dlimbs_div_estimate(uint32_t q[DLIMBS], const uint32_t x[DLIMBS],
                      const uint32_t approx[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t, c;
    uint64_t w;

    // computes q=(x*approx>>32*LIMBS) + x + 3
    //          q=min(q, (1<<32*LIMBS)-1);
    //
    // Notes:   leaves junk in lower words of q

    w = 0;
#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, x[0], TPI - LIMBS + index, TPI);
      w = mad_wide(t, approx[0], w);
      t = __shfl_sync(sync, ulow(w), threadIdx.x + 1,
                      TPI); // half size: take advantage of zero wrapping
      w = (w >> 32) + t;
    }

    // increase the estimate by 3
    t = (group_thread == TPI - LIMBS) ? 3 : 0;
    w = w + t + x[0];

    q[0] = ulow(w);
    c = uhigh(w);
    if (core::resolve_add(c, q[0]) != 0)
      q[0] = 0xFFFFFFFF;
  }

  __device__ __forceinline__ static void
  dlimbs_sqrt_estimate(uint32_t q[DLIMBS], uint32_t top,
                       const uint32_t x[DLIMBS],
                       const uint32_t approx[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t, high, low;
    uint64_t w;

    // computes:
    //    1.  num=((top<<32*LIMBS) + x) / 2
    //    2.  q=(num*approx>>32*LIMBS) + num + 4
    //    3.  q=min(q, (1<<32*LIMBS)-1);
    //
    //  Note:  Leaves junk in lower words of q

    // shift x right by 1 bit.  Fill high bit with top.
    t = __shfl_down_sync(sync, x[0], 1, TPI);
    t = (group_thread == TPI - 1) ? top : t;
    low = uright_wrap(x[0], t, 1);

    // if we're exactly half the size, need to clear out low limb
    if (TPI == 2 * LIMBS)
      low = (group_thread >= LIMBS) ? low : 0;

    // estimate is in low
    w = 0;
#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, low, TPI - LIMBS + index, TPI);
      w = mad_wide(t, approx[0], w);
      t = __shfl_sync(sync, ulow(w), threadIdx.x + 1,
                      TPI); // half size: take advantage of zero wrapping
      w = (w >> 32) + t;
    }

    // increase the estimate by 4 -- because we might have cleared low bit,
    // estimate can be off by 4
    t = (group_thread == TPI - LIMBS) ? 4 : 0;
    w = w + t + low;

    low = ulow(w);
    high = uhigh(w);
    if (core::resolve_add(high, low) != 0)
      low = 0xFFFFFFFF;
    q[0] = low;
  }
};

template <class core> class dispatch_dlimbs_t<core, dlimbs_algs_full> {
public:
  static const uint32_t TPI = core::TPI;
  static const uint32_t LIMBS = core::LIMBS;
  static const uint32_t DLIMBS = core::DLIMBS;
  static const uint32_t LIMB_OFFSET = DLIMBS * TPI - LIMBS;

  // These algorithms are used then LIMBS<=TPI.  Almost the same as the half
  // size ones, few tweaks here and there.

  __device__ __forceinline__ static void
  dlimbs_approximate(uint32_t approx[DLIMBS], const uint32_t denom[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t x, d0, d1, x0, x1, x2, est, a, h, l;
    int32_t c, top;

    // computes (beta^2 - 1) / denom - beta, where beta=1<<32*LIMBS

    x = 0xFFFFFFFF - denom[0];

    d1 = __shfl_sync(sync, denom[0], TPI - 1, TPI);
    d0 = __shfl_sync(sync, denom[0], TPI - 2, TPI);

    approx[0] = 0;
    a = uapprox(d1);

#pragma nounroll
    for (int32_t thread = LIMBS - 1; thread >= 0; thread--) {
      x0 = __shfl_sync(sync, x, TPI - 3, TPI);
      x1 = __shfl_sync(sync, x, TPI - 2, TPI);
      x2 = __shfl_sync(sync, x, TPI - 1, TPI);
      est = udiv(x0, x1, x2, d0, d1, a);

      l = madlo_cc(est, denom[0], 0);
      h = madhic(est, denom[0], 0);

      x = sub_cc(x, h);
      c = subc(0, 0); // thread TPI-1 is zero

      top = __shfl_sync(sync, x, TPI - 1, TPI);
      x = __shfl_up_sync(sync, x, 1, TPI);
      c = __shfl_sync(sync, c, threadIdx.x - 1, TPI);
      x = (group_thread == 0) ? 0xFFFFFFFF : x;

      x = sub_cc(x, l);
      c = subc(c, 0);

      if (top + core::resolve_sub(c, x) < 0) {
        // means a correction is required, should be very rare
        x = add_cc(x, denom[0]);
        c = addc(0, 0);
        core::fast_propagate_add(c, x);
        est--;
      }
      approx[0] = (group_thread == thread + TPI - LIMBS) ? est : approx[0];
    }
  }

  __device__ __forceinline__ static uint32_t
  dlimbs_sqrt_rem_wide(uint32_t s[DLIMBS], uint32_t r[DLIMBS],
                       const uint32_t lo[DLIMBS], const uint32_t hi[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t x, x0, x1, t0, t1, divisor, approx, p, q, c, low;

    // computes s=sqrt(x), r=x-s^2, where x=(hi<<32*LIMBS) + lo

    low = lo[0];
    x = hi[0];
    if (TPI != LIMBS) {
      low = __shfl_sync(sync, low, threadIdx.x - TPI + LIMBS, TPI);
      x = ((int32_t)group_thread >= (int32_t)(TPI - LIMBS))
              ? x
              : low; // use casts to silence warning
    }
    x0 = __shfl_sync(sync, x, TPI - 2, TPI);
    x1 = __shfl_sync(sync, x, TPI - 1, TPI);

    divisor = usqrt(x0, x1);
    approx = uapprox(divisor);

    t0 = madlo_cc(divisor, divisor, 0);
    t1 = madhic(divisor, divisor, 0);
    x0 = sub_cc(x0, t0);
    x1 = subc(x1, t1);

    x = (group_thread == TPI - 1) ? low : x;
    x = __shfl_sync(sync, x, threadIdx.x - 1, TPI);
    x = (group_thread == TPI - 1) ? x0 : x;
    s[0] = (group_thread == TPI - 1) ? divisor + divisor
                                     : 0; // silent 1 at the top of s

#pragma nounroll
    for (int32_t index = TPI - 2; index >= (int32_t)(TPI - LIMBS); index--) {
      x0 = __shfl_sync(sync, x, TPI - 1, TPI);
      q = usqrt_div(x0, x1, divisor, approx);
      s[0] = (group_thread == index) ? q : s[0];

      p = madhi(q, s[0], 0);
      x = sub_cc(x, p);
      c = subc(0, 0);
      core::fast_propagate_sub(c, x);

      x1 = __shfl_sync(sync, x, TPI - 1, TPI) -
           q; // we subtract q because of the silent 1 at the top of s
      t0 = __shfl_sync(sync, low, index, TPI);
      x = __shfl_up_sync(sync, x, 1, TPI);
      x = (group_thread == 0) ? t0 : x;

      p = madlo(q, s[0], 0);
      x = sub_cc(x, p);
      c = subc(0, 0);
      x1 -= core::fast_propagate_sub(c, x);

      while (0 > (int32_t)x1) {
        x1++;
        q--;

        // correction step: add q and s
        x = add_cc(x, (group_thread == index) ? q : 0);
        c = addc(0, 0);
        x = add_cc(x, s[0]);
        c = addc(c, 0);

        x1 += core::resolve_add(c, x);

        // update s
        s[0] = (group_thread == index) ? q : s[0];
      }
      s[0] = (group_thread == index + 1) ? s[0] + (q >> 31) : s[0];
      s[0] = (group_thread == index) ? q + q : s[0];
    }
    t0 = __shfl_down_sync(sync, s[0], 1, TPI);
    t0 = (group_thread == TPI - 1) ? 1 : t0;
    s[0] = uright_wrap(s[0], t0, 1);
    r[0] = x;
    return x1;
  }

  __device__ __forceinline__ static void
  dlimbs_div_estimate(uint32_t q[DLIMBS], const uint32_t x[DLIMBS],
                      const uint32_t approx[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t, c;
    uint64_t w;

    // computes q=(x*approx>>32*LIMBS) + x + 3
    //          q=min(q, (1<<32*LIMBS)-1);
    //
    // Notes:   leaves junk in lower words of q

    w = 0;
#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, x[0], TPI - LIMBS + index, TPI);
      w = mad_wide(t, approx[0], w);
      t = __shfl_sync(sync, ulow(w), threadIdx.x + 1, TPI);
      t = (group_thread == TPI - 1) ? 0 : t;
      w = (w >> 32) + t;
    }

    // increase the estimate by 3
    t = (group_thread == TPI - LIMBS) ? 3 : 0;
    w = w + t + x[0];

    q[0] = ulow(w);
    c = uhigh(w);
    if (core::resolve_add(c, q[0]) != 0)
      q[0] = 0xFFFFFFFF;
  }

  __device__ __forceinline__ static void
  dlimbs_sqrt_estimate(uint32_t q[DLIMBS], uint32_t top,
                       const uint32_t x[DLIMBS],
                       const uint32_t approx[DLIMBS]) {
    uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & TPI - 1;
    uint32_t t, high, low;
    uint64_t w;

    // computes:
    //    1.  num=((top<<32*LIMBS) + x) / 2
    //    2.  q=(num*approx>>32*LIMBS) + num + 4
    //    3.  q=min(q, (1<<32*LIMBS)-1);
    //
    //  Note:  Leaves junk in lower words of q

    // shift x right by 1 bit.  Fill high bit with top.
    t = __shfl_down_sync(sync, x[0], 1, TPI);
    t = (group_thread == TPI - 1) ? top : t;
    low = uright_wrap(x[0], t, 1);

    // estimate is in low
    w = 0;
#pragma unroll
    for (int32_t index = 0; index < LIMBS; index++) {
      t = __shfl_sync(sync, low, TPI - LIMBS + index, TPI);
      w = mad_wide(t, approx[0], w);
      t = __shfl_down_sync(sync, ulow(w), 1, TPI);
      t = (group_thread == TPI - 1) ? 0 : t;
      w = (w >> 32) + t;
    }

    // increase the estimate by 4 -- because we might have cleared low bit,
    // estimate can be off by 4
    t = (group_thread == TPI - LIMBS) ? 4 : 0;
    w = w + t + low;

    low = ulow(w);
    high = uhigh(w);
    if (core::resolve_add(high, low) != 0)
      low = 0xFFFFFFFF;
    q[0] = low;
  }
};

template <class core> class dispatch_dlimbs_t<core, dlimbs_algs_multi> {
public:
  static const uint32_t TPI = core::TPI;
  static const uint32_t LIMBS = core::LIMBS;
  static const uint32_t DLIMBS = core::DLIMBS;
  static const uint32_t LIMB_OFFSET = DLIMBS * TPI - LIMBS;

  // These algorithms are used when LIMBS>TPI, i.e. the reciprocal window spans
  // more than one word per thread (DLIMBS>=2).  Rather than distribute the
  // Newton-Raphson reciprocal across the warp (as the half/full variants do for
  // DLIMBS==1), each thread gathers the full LIMBS-word dlimbs quantity and
  // evaluates the same well-defined integer contract serially and redundantly,
  // then keeps its own DLIMBS-word share of the result.  The contracts are
  // exactly those documented for the half/full variants:
  //
  //   dlimbs_approximate     : approx = floor((beta^2-1)/denom) - beta,
  //   beta=2^(32*LIMBS) dlimbs_div_estimate    : q = min(beta-1,
  //   high_word(x*approx) + x + 3) dlimbs_sqrt_rem_wide   :
  //   x=(hi<<32*LIMBS)+lo; s=isqrt(x); r=x-s^2 (returns top word of r)
  //   dlimbs_sqrt_estimate   : num=((top<<32*LIMBS)+x)/2; q=min(beta-1,
  //   high_word(num*approx)+num+4)
  //
  // The serial helpers below operate on little-endian uint32 arrays.  All loops
  // are marked "#pragma unroll 1" because LIMBS can be large (up to 256 at
  // TPI=32/256K bits) and unrolling would explode code size.

  __device__ static int32_t smp_cmp(const uint32_t* a, const uint32_t* b,
                                    int32_t n) {
#pragma unroll 1
    for (int32_t i = n - 1; i >= 0; i--) {
      if (a[i] < b[i])
        return -1;
      if (a[i] > b[i])
        return 1;
    }
    return 0;
  }
  __device__ static uint32_t smp_sub(uint32_t* r, const uint32_t* a,
                                     const uint32_t* b, int32_t n) {
    uint64_t borrow = 0;
#pragma unroll 1
    for (int32_t i = 0; i < n; i++) {
      uint64_t t = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
      r[i] = (uint32_t)t;
      borrow = (t >> 63) & 0x1;
    }
    return (uint32_t)borrow;
  }
  __device__ static uint32_t smp_add(uint32_t* r, const uint32_t* a,
                                     const uint32_t* b, int32_t n) {
    uint64_t carry = 0;
#pragma unroll 1
    for (int32_t i = 0; i < n; i++) {
      uint64_t t = (uint64_t)a[i] + (uint64_t)b[i] + carry;
      r[i] = (uint32_t)t;
      carry = t >> 32;
    }
    return (uint32_t)carry;
  }
  __device__ static void smp_shl1(uint32_t* a, int32_t n) {
    uint32_t carry = 0;
#pragma unroll 1
    for (int32_t i = 0; i < n; i++) {
      uint32_t nc = a[i] >> 31;
      a[i] = (a[i] << 1) | carry;
      carry = nc;
    }
  }
  // full LIMBSxLIMBS -> 2*LIMBS product, split into hi[LIMBS]/lo[LIMBS]
  __device__ static void smp_mul(uint32_t* hi, uint32_t* lo, const uint32_t* a,
                                 const uint32_t* b) {
    uint32_t p[2 * LIMBS];
#pragma unroll 1
    for (int32_t i = 0; i < 2 * (int32_t)LIMBS; i++)
      p[i] = 0;
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++) {
      uint64_t carry = 0;
#pragma unroll 1
      for (int32_t j = 0; j < (int32_t)LIMBS; j++) {
        uint64_t t =
            (uint64_t)a[i] * (uint64_t)b[j] + (uint64_t)p[i + j] + carry;
        p[i + j] = (uint32_t)t;
        carry = t >> 32;
      }
      p[i + LIMBS] = (uint32_t)((uint64_t)p[i + LIMBS] + carry);
    }
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++) {
      lo[i] = p[i];
      hi[i] = p[i + LIMBS];
    }
  }

  // gather a dlimbs quantity so every thread holds the full LIMBS-word value
  __device__ __forceinline__ static void gather(uint32_t full[LIMBS],
                                                const uint32_t x[DLIMBS]) {
    dispatch_dlimbs_t<core, dlimbs_algs_common>::dlimbs_all_gather(full, x);
  }
  // keep this thread's DLIMBS-word share of a full LIMBS-word value (local, no
  // shuffle). inverse of the common scatter/gather layout: word `index` lives
  // at thread (index+LIMB_OFFSET)/DLIMBS, dlimb (index+LIMB_OFFSET)%DLIMBS.
  __device__ __forceinline__ static void scatter(uint32_t out[DLIMBS],
                                                 const uint32_t full[LIMBS]) {
    uint32_t group_thread = threadIdx.x & TPI - 1;
#pragma unroll
    for (int32_t j = 0; j < (int32_t)DLIMBS; j++) {
      int32_t index =
          (int32_t)(group_thread * DLIMBS) + j - (int32_t)LIMB_OFFSET;
      out[j] = (index >= 0 && index < (int32_t)LIMBS) ? full[index] : 0;
    }
  }

  __device__ __forceinline__ static void
  dlimbs_approximate(uint32_t approx[DLIMBS], const uint32_t denom[DLIMBS]) {
    uint32_t D[LIMBS], A[LIMBS], rem[LIMBS + 1], Dp[LIMBS + 1];

    // approx = floor((beta^2-1)/denom) - beta, computed by binary long division
    // of the 2*LIMBS-word all-ones numerator by the normalized denom.  Since
    // denom has its top bit set, the quotient lies in (beta, 2*beta), so
    // approx = quotient mod beta = the low LIMBS words we accumulate.
    gather(D, denom);
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++) {
      Dp[i] = D[i];
      A[i] = 0;
      rem[i] = 0;
    }
    Dp[LIMBS] = 0;
    rem[LIMBS] = 0;
#pragma unroll 1
    for (int32_t k = 64 * (int32_t)LIMBS - 1; k >= 0; k--) {
      smp_shl1(rem, LIMBS + 1);
      rem[0] |= 1u;       // every bit of the numerator is 1
      smp_shl1(A, LIMBS); // drop overflow above LIMBS words
      if (smp_cmp(rem, Dp, LIMBS + 1) >= 0) {
        smp_sub(rem, rem, Dp, LIMBS + 1);
        A[0] |= 1u;
      }
    }
    scatter(approx, A);
  }

  __device__ __forceinline__ static void
  dlimbs_div_estimate(uint32_t q[DLIMBS], const uint32_t x[DLIMBS],
                      const uint32_t approx[DLIMBS]) {
    uint32_t X[LIMBS], AP[LIMBS], hi[LIMBS], lo[LIMBS], Q[LIMBS];
    uint32_t carry;
    uint64_t t, c;

    gather(X, x);
    gather(AP, approx);
    smp_mul(hi, lo, X, AP);           // hi = high LIMBS words of x*approx
    carry = smp_add(Q, hi, X, LIMBS); // Q = hi + x
    t = (uint64_t)Q[0] + 3;           // + 3
    Q[0] = (uint32_t)t;
    c = t >> 32;
#pragma unroll 1
    for (int32_t i = 1; i < (int32_t)LIMBS && c != 0; i++) {
      t = (uint64_t)Q[i] + c;
      Q[i] = (uint32_t)t;
      c = t >> 32;
    }
    carry += (uint32_t)c;
    if (carry != 0) { // clamp to beta-1
#pragma unroll 1
      for (int32_t i = 0; i < (int32_t)LIMBS; i++)
        Q[i] = 0xFFFFFFFFu;
    }
    scatter(q, Q);
  }

  __device__ __forceinline__ static uint32_t
  dlimbs_sqrt_rem_wide(uint32_t s[DLIMBS], uint32_t r[DLIMBS],
                       const uint32_t lo[DLIMBS], const uint32_t hi[DLIMBS]) {
    uint32_t LO[LIMBS], HI[LIMBS], X[2 * LIMBS], rem[LIMBS + 1], S[LIMBS],
        b[LIMBS + 1], R[LIMBS];
    int32_t N = 32 * (int32_t)LIMBS;

    // x = (hi<<32*LIMBS) + lo, a 2*LIMBS-word value; produce s=isqrt(x) and
    // remainder x-s^2 by the classic bit-pair (restoring) integer square root.
    gather(LO, lo);
    gather(HI, hi);
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++) {
      X[i] = LO[i];
      X[i + LIMBS] = HI[i];
      S[i] = 0;
      rem[i] = 0;
    }
    rem[LIMBS] = 0;
#pragma unroll 1
    for (int32_t i = N - 1; i >= 0; i--) {
      smp_shl1(rem, LIMBS + 1);
      smp_shl1(rem, LIMBS + 1); // rem <<= 2
      int32_t bitpos = 2 * i;
      uint32_t two = (X[bitpos >> 5] >> (bitpos & 31)) &
                     0x3u; // next two bits, MSB pair first
      rem[0] |= two;
#pragma unroll 1
      for (int32_t t = 0; t < (int32_t)LIMBS; t++) // b = 4*S + 1
        b[t] = S[t];
      b[LIMBS] = 0;
      smp_shl1(b, LIMBS + 1);
      smp_shl1(b, LIMBS + 1);
      b[0] |= 1u;
      smp_shl1(S, LIMBS); // S = 2*S
      if (smp_cmp(rem, b, LIMBS + 1) >= 0) {
        smp_sub(rem, rem, b, LIMBS + 1);
        S[0] |= 1u;
      }
    }
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++)
      R[i] = rem[i];
    scatter(s, S);
    scatter(r, R);
    return rem[LIMBS]; // top word of the remainder
  }

  __device__ __forceinline__ static void
  dlimbs_sqrt_estimate(uint32_t q[DLIMBS], uint32_t top,
                       const uint32_t x[DLIMBS],
                       const uint32_t approx[DLIMBS]) {
    uint32_t X[LIMBS], AP[LIMBS], num[LIMBS], hi[LIMBS], lo[LIMBS], Q[LIMBS];
    uint32_t carry;
    uint64_t t, c;

    gather(X, x);
    gather(AP, approx);
// num = ((top<<32*LIMBS) + x) >> 1
#pragma unroll 1
    for (int32_t i = 0; i < (int32_t)LIMBS; i++) {
      uint32_t hibit = (i + 1 < (int32_t)LIMBS) ? (X[i + 1] & 1u) : (top & 1u);
      num[i] = (X[i] >> 1) | (hibit << 31);
    }
    smp_mul(hi, lo, num, AP);           // hi = high LIMBS words of num*approx
    carry = smp_add(Q, hi, num, LIMBS); // Q = hi + num
    t = (uint64_t)Q[0] + 4;             // + 4
    Q[0] = (uint32_t)t;
    c = t >> 32;
#pragma unroll 1
    for (int32_t i = 1; i < (int32_t)LIMBS && c != 0; i++) {
      t = (uint64_t)Q[i] + c;
      Q[i] = (uint32_t)t;
      c = t >> 32;
    }
    carry += (uint32_t)c;
    if (carry != 0) { // clamp to beta-1
#pragma unroll 1
      for (int32_t i = 0; i < (int32_t)LIMBS; i++)
        Q[i] = 0xFFFFFFFFu;
    }
    scatter(q, Q);
  }
};

} /* namespace cgbn */