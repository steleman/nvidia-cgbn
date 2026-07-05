CGBN Beta Release (October 2018)

This beta release has several important improvements over the alpha release:

*  Support for threads per instance (TPI) of 4, 8, and 16 in addition to the original 32
*  Support for all sizes from 32 bits to 32K bits, provided the size is evenly divisible by 32
*  Performance improvements when for sizes<2K bits, with TPI=8
*  C style wrappers for all cgbn_env_t methods


Minor update (June 2021)

*  Added build tags for turing and ampere
*  Fixed the depricated gtest TEST_CASE_P warnings


Signed integer support (February 2025)

*  Added signed (two's complement) integer operations in cgbn/cgbn_signed.h, using the same fixed-width
   encoding as the existing unsigned integers.  New cgbn_signed_* wrappers cover the operations whose result
   depends on the sign: compare, arithmetic (sign-extending) right shift, truncated (round-toward-zero)
   division and remainder, high-half and double-width products, absolute value, and 32-bit signed set/get.
   All other operations (add, sub, negate, low-half multiply, bitwise, shifts left, load/store, ...) are
   bit-identical for signed and unsigned values and reuse the existing unsigned API.
*  Added unit tests (unit_tests suite CGBN6) and a worked example (samples/sample_05_signed).


Maximum size raised to 256K bits (March 2025)

*  Raised the supported size range from 32K bits to 256K bits (up to 262144 bits, still evenly divisible
   by 32).  Add, subtract, multiply, shift, rotate, bitwise, compare and Montgomery operations already
   scaled to any size; the 32K ceiling was actually the divide/square-root family, which above 32K
   (LIMBS > TPI) dispatched to the previously-unimplemented "dlimbs_algs_multi" reciprocal.
*  Implemented dispatch_dlimbs_t<core, dlimbs_algs_multi> in cgbn/core/dispatch_dlimbs.cu.  For the
   multi-word regime each thread gathers the full LIMBS-word reciprocal window and evaluates the same
   documented integer contract (reciprocal, quotient estimate, integer square root, sqrt estimate)
   serially, then keeps its DLIMBS-word share -- so division, remainder, square root and the Barrett
   routines now work up to 256K bits.  Correctness was validated against GMP on GPU for div/rem/div_rem,
   sqrt/sqrt_rem, the wide variants and Barrett, across TPI 4/8/16/32 and DLIMBS 2/3/4/8.
*  Added size classes size65536t32, size131072t32, size262144t32 (unit_tests/sizes.h) and wired the
   FULL_TEST instantiations for all suites.
*  Added a worked example (samples/sample_06_large_div) computing 256K-bit division, remainder and
   square root with GMP verification.
*  Fixed a pre-existing rotate bug: the unpadded (PADDING==0) cgbn_rotate_left/cgbn_rotate_right passed
   the raw rotation amount to the distributed rotate, which feeds numbits>>5 to static_divide_small (only
   exact over a bounded range).  A rotation amount whose reduced value was small but whose raw value was
   large (e.g. 2147483680 == 32 mod 262144) could rotate incorrectly at large limb counts.  The amount is
   now reduced modulo BITS first, matching the padded path.  Surfaced by the 256K unit tests.
*  Should be possible to raise the maximum size to 512K bits.
*  Tested on a QUADRO RTX A6000 (sm_80).

Ada support (August 2025)

*  Added support for Ada architecture (sm_89). Tested on a GeForce RTX 4090 FE.

Blackwell support (May 2026)

*  Added support for Blackwell architecture (sm_120). Tested on an RTX 4500 Pro Blackwell.


