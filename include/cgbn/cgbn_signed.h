/***

Copyright (c) 2024-2026, Stefan Teleman. All rights reserved.

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

/****************************************************************************************************************
 *
 * Signed (two's complement) integer support for CGBN.
 *
 * CGBN stores an unsigned value as a fixed BITS-wide, little-endian array of 32-bit limbs.  A *signed* CGBN
 * uses the identical encoding, reinterpreted as two's complement: the most significant bit (bit BITS-1) is the
 * sign bit.  Because CGBN uses a fixed width, the following operations are bit-for-bit identical for signed and
 * unsigned values and therefore have NO separate signed variant -- just use the existing unsigned routine:
 *
 *     cgbn_set, cgbn_swap, cgbn_add, cgbn_sub, cgbn_negate, cgbn_mul (low half), cgbn_sqr (low half),
 *     cgbn_equals, cgbn_shift_left, cgbn_rotate_left, cgbn_rotate_right, all cgbn_bitwise_* routines,
 *     and cgbn_load / cgbn_store.
 *
 * Only the operations whose result depends on the sign interpretation get a cgbn_signed_* wrapper below:
 * comparison, arithmetic (sign-extending) right shift, truncated division / remainder, and the high-half /
 * double-width products.  Every wrapper is implemented purely in terms of the public unsigned cgbn_* API, so
 * it works unchanged on every CGBN backend (CUDA device, GMP/mpz host reference, and CPU).
 *
 * Division and remainder truncate toward zero (C / GMP mpz_tdiv semantics): the quotient truncates toward
 * zero and the remainder takes the sign of the dividend, so that  q*denom + r == num  and  |r| < |denom|.
 *
 ****************************************************************************************************************/

/* sign inspection / absolute value */

// returns true if the two's complement value of a is negative (its top bit is set)
template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_signed_is_negative(env_t env, const typename env_t::cgbn_t &a) {
  return cgbn_extract_bits_ui32(env, a, env_t::BITS-1, 1)!=0;
}

// negation is identical to the unsigned two's complement negation; provided for naming symmetry.
// returns -1 when a overflows (a == the most negative value), otherwise 0, matching cgbn_negate.
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_signed_negate(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  return cgbn_negate(env, r, a);
}

// stores |a| into r and returns true if a was negative.  Note: abs(most-negative) overflows back to itself.
template<class env_t>
__host__ __device__ __forceinline__ bool cgbn_signed_abs(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  if(cgbn_signed_is_negative(env, a)) {
    cgbn_negate(env, r, a);
    return true;
  }
  cgbn_set(env, r, a);
  return false;
}


/* small signed integer set / get */

// sets r to the sign-extended value of a signed 32-bit integer
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_set_i32(env_t env, typename env_t::cgbn_t &r, const int32_t value) {
  cgbn_set_ui32(env, r, (uint32_t)value);
  if(value<0 && env_t::BITS>32)
    cgbn_bitwise_mask_ior(env, r, r, -(int32_t)(env_t::BITS-32));   // fill the top BITS-32 bits with the sign
}

// returns the least significant 32 bits of a, reinterpreted as a signed 32-bit integer
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_signed_get_i32(env_t env, const typename env_t::cgbn_t &a) {
  return (int32_t)cgbn_get_ui32(env, a);
}


/* comparison */

// signed compare: returns 1 if a>b, 0 if a==b, and -1 if a<b (two's complement ordering)
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_signed_compare(env_t env, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  bool a_neg=cgbn_signed_is_negative(env, a);
  bool b_neg=cgbn_signed_is_negative(env, b);

  // if the signs differ, the negative value is the smaller one
  if(a_neg!=b_neg)
    return a_neg ? -1 : 1;
  // same sign: two's complement ordering agrees with unsigned ordering
  return cgbn_compare(env, a, b);
}

// signed compare against a sign-extended 32-bit constant
template<class env_t>
__host__ __device__ __forceinline__ int32_t cgbn_signed_compare_i32(env_t env, const typename env_t::cgbn_t &a, const int32_t value) {
  typename env_t::cgbn_t v;

  cgbn_signed_set_i32(env, v, value);
  return cgbn_signed_compare(env, a, v);
}


/* arithmetic (sign-extending) right shift */

// r = a >> numbits, replicating the sign bit into the vacated high bits
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_shift_right(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const uint32_t numbits) {
  bool neg=cgbn_signed_is_negative(env, a);

  if(numbits>=env_t::BITS) {
    // shifting by the full width (or more) yields all-sign: 0 for non-negative, -1 for negative
    if(neg)
      cgbn_bitwise_mask_copy(env, r, env_t::BITS);   // all ones == -1
    else
      cgbn_set_ui32(env, r, 0);
    return;
  }

  cgbn_shift_right(env, r, a, numbits);              // logical shift
  if(neg && numbits>0)
    cgbn_bitwise_mask_ior(env, r, r, -(int32_t)numbits);   // set the top numbits bits to the sign
}


/* truncated division and remainder (round toward zero) */

// q = trunc(num / denom)
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_div(env_t env, typename env_t::cgbn_t &q, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  typename env_t::cgbn_t abs_num, abs_denom;
  bool                   num_neg, denom_neg;

  num_neg=cgbn_signed_abs(env, abs_num, num);
  denom_neg=cgbn_signed_abs(env, abs_denom, denom);
  cgbn_div(env, q, abs_num, abs_denom);
  if(num_neg!=denom_neg)
    cgbn_negate(env, q, q);
}

// r = num - trunc(num / denom) * denom  (r takes the sign of num)
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_rem(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  typename env_t::cgbn_t abs_num, abs_denom;
  bool                   num_neg;

  num_neg=cgbn_signed_abs(env, abs_num, num);
  cgbn_signed_abs(env, abs_denom, denom);
  cgbn_rem(env, r, abs_num, abs_denom);
  if(num_neg)
    cgbn_negate(env, r, r);
}

// q = trunc(num / denom), r = num - q * denom
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_div_rem(env_t env, typename env_t::cgbn_t &q, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &num, const typename env_t::cgbn_t &denom) {
  typename env_t::cgbn_t abs_num, abs_denom;
  bool                   num_neg, denom_neg;

  num_neg=cgbn_signed_abs(env, abs_num, num);
  denom_neg=cgbn_signed_abs(env, abs_denom, denom);
  cgbn_div_rem(env, q, r, abs_num, abs_denom);
  if(num_neg!=denom_neg)
    cgbn_negate(env, q, q);
  if(num_neg)
    cgbn_negate(env, r, r);
}


/* high-half and double-width signed products */

// r = high BITS bits of the signed product a * b
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_mul_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  // signed_high = unsigned_high - (a<0 ? b : 0) - (b<0 ? a : 0)
  cgbn_mul_high(env, r, a, b);
  if(cgbn_signed_is_negative(env, a))
    cgbn_sub(env, r, r, b);
  if(cgbn_signed_is_negative(env, b))
    cgbn_sub(env, r, r, a);
}

// r = high BITS bits of the signed product a * a
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_sqr_high(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a) {
  cgbn_sqr_high(env, r, a);
  if(cgbn_signed_is_negative(env, a)) {
    cgbn_sub(env, r, r, a);
    cgbn_sub(env, r, r, a);
  }
}

// r = full 2*BITS signed product a * b  (r._low holds the low BITS bits, r._high the high BITS bits)
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_mul_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b) {
  // the sign correction only touches the high word (it subtracts multiples of 2^BITS); the low word is exact
  cgbn_mul_wide(env, r, a, b);
  if(cgbn_signed_is_negative(env, a))
    cgbn_sub(env, r._high, r._high, b);
  if(cgbn_signed_is_negative(env, b))
    cgbn_sub(env, r._high, r._high, a);
}

// r = full 2*BITS signed product a * a
template<class env_t>
__host__ __device__ __forceinline__ void cgbn_signed_sqr_wide(env_t env, typename env_t::cgbn_wide_t &r, const typename env_t::cgbn_t &a) {
  cgbn_sqr_wide(env, r, a);
  if(cgbn_signed_is_negative(env, a)) {
    cgbn_sub(env, r._high, r._high, a);
    cgbn_sub(env, r._high, r._high, a);
  }
}
