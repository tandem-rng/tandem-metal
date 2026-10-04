/* Prints the cross-check fixtures of tandem-c and tandem-cuda as JSON, see gen_fixtures.sh.
 * 64-bit values and the bits of floats are hex strings, so that every value survives exactly. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "cross_below.h"
#include "cross_exponential.h"
#include "cross_fill_below.h"
#include "cross_normal.h"
#include "cuda/cross_fill_below.h"
#include "cuda/cross_fill_exponential.h"
#include "cuda/cross_fill_normal.h"

#define LEN(a) (sizeof(a) / sizeof((a)[0]))

static void u32s(const uint32_t *v, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%s%" PRIu32, i ? ", " : "[", v[i]);
    printf("]");
}

static void u64s(const uint64_t *v, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%s\"%016" PRIx64 "\"", i ? ", " : "[", v[i]);
    printf("]");
}

static void f64s(const double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint64_t b;
        memcpy(&b, &v[i], 8);
        printf("%s\"%016" PRIx64 "\"", i ? ", " : "[", b);
    }
    printf("]");
}

static void f32s(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &v[i], 4);
        printf("%s%" PRIu32, i ? ", " : "[", b);
    }
    printf("]");
}

/* One array of row objects: `fields` prints the row's fields, the loop the separators. */
#define ROWS(name, arr, fields)                                                                  \
    do {                                                                                         \
        printf("  \"%s\": [\n", name);                                                           \
        for (size_t c = 0; c < LEN(arr); c++) {                                                  \
            printf("    {");                                                                     \
            fields;                                                                              \
            printf("}%s\n", c + 1 < LEN(arr) ? "," : "");                                        \
        }                                                                                        \
        printf("  ],\n");                                                                        \
    } while (0)

int main(void) {
    printf("{\n  \"key\": ");
    u32s(CROSS_FILL_KEY, 4);
    printf(",\n");

    /* Scalar bounded draws from the key at position 1 (one bool draw after seeding). */
    ROWS("below_u32", CROSS_U32, printf("\"range\": %" PRIu32 ", \"end\": %" PRIu64 ", \"values\": ",
                                        CROSS_U32[c].n, CROSS_U32[c].end_pos);
         u32s(CROSS_U32[c].want, CROSS_COUNT));
    ROWS("below_u64", CROSS_U64, printf("\"range\": \"%016" PRIx64 "\", \"end\": %" PRIu64 ", \"values\": ",
                                        CROSS_U64[c].n, CROSS_U64[c].end_pos);
         u64s(CROSS_U64[c].want, CROSS_COUNT));

    /* Bounded fills of 64 elements: tandem-c's at its starts, tandem-cuda's at 0 and its starts. */
    ROWS("fill_below_u32", CROSS_FILL_U32,
         printf("\"start\": %" PRIu64 ", \"range\": %" PRIu32 ", \"end\": %" PRIu64 ", \"values\": ",
                CROSS_FILL_U32[c].start, CROSS_FILL_U32[c].n, CROSS_FILL_U32[c].end_pos);
         u32s(CROSS_FILL_U32[c].want, CROSS_COUNT));
    ROWS("fill_below_u64", CROSS_FILL_U64,
         printf("\"start\": %" PRIu64 ", \"range\": \"%016" PRIx64 "\", \"end\": %" PRIu64 ", \"values\": ",
                CROSS_FILL_U64[c].start, CROSS_FILL_U64[c].n, CROSS_FILL_U64[c].end_pos);
         u64s(CROSS_FILL_U64[c].want, CROSS_COUNT));
    ROWS("cuda_below_u32", CROSS_BELOW32,
         printf("\"start\": 0, \"range\": %" PRIu32 ", \"rejected\": %u, \"values\": ", CROSS_BELOW32[c].range,
                CROSS_BELOW32[c].rejected);
         u32s(CROSS_BELOW32[c].out, 64));
    ROWS("cuda_below_u64", CROSS_BELOW64,
         printf("\"start\": 0, \"range\": \"%016" PRIx64 "\", \"rejected\": %u, \"values\": ",
                CROSS_BELOW64[c].range, CROSS_BELOW64[c].rejected);
         u64s(CROSS_BELOW64[c].out, 64));
    ROWS("cuda_below_u32_at", CROSS_BELOW32_AT,
         printf("\"start\": %" PRIu64 ", \"range\": %" PRIu32 ", \"rejected\": %u, \"values\": ",
                CROSS_BELOW32_AT[c].start, CROSS_BELOW32_AT[c].range, CROSS_BELOW32_AT[c].rejected);
         u32s(CROSS_BELOW32_AT[c].out, 64));
    ROWS("cuda_below_u64_at", CROSS_BELOW64_AT,
         printf("\"start\": %" PRIu64 ", \"range\": \"%016" PRIx64 "\", \"rejected\": %u, \"values\": ",
                CROSS_BELOW64_AT[c].start, CROSS_BELOW64_AT[c].range, CROSS_BELOW64_AT[c].rejected);
         u64s(CROSS_BELOW64_AT[c].out, 64));

    /* Ziggurat fills of 64 at tandem-c's starts, Box-Muller pairs from position 1, then normal
     * fills at tandem-cuda's start positions. */
    ROWS("normal_f64", CROSS_NORMAL,
         printf("\"start\": %" PRIu64 ", \"end\": %" PRIu64 ", \"values\": ", CROSS_NORMAL[c].start,
                CROSS_NORMAL[c].end_pos);
         f64s(CROSS_NORMAL[c].want, CROSS_NORMAL_COUNT));
    printf("  \"normal_f32\": {\"end\": %" PRIu64 ", \"values\": ", CROSS_NORMALF_END_POS);
    f32s(CROSS_NORMALF, 2 * CROSS_NORMAL_COUNT);
    printf("},\n");
    ROWS("fill_normal_f64", CROSS_NORMAL64,
         printf("\"start\": %" PRIu64 ", \"n\": %u, \"values\": ", CROSS_NORMAL64[c].pos, CROSS_NORMAL64[c].n);
         f64s(CROSS_NORMAL64[c].out, CROSS_NORMAL64[c].n));
    ROWS("fill_normal_f32", CROSS_NORMAL32,
         printf("\"start\": %" PRIu64 ", \"n\": %u, \"values\": ", CROSS_NORMAL32[c].pos, CROSS_NORMAL32[c].n);
         f32s(CROSS_NORMAL32[c].out, CROSS_NORMAL32[c].n));

    /* Exponential fills of 64 from tandem-c, then tandem-cuda's at start positions. */
    ROWS("exponential_f64", CROSS_EXPONENTIAL,
         printf("\"start\": %" PRIu64 ", \"end\": %" PRIu64 ", \"values\": ", CROSS_EXPONENTIAL[c].start,
                CROSS_EXPONENTIAL[c].end_pos);
         f64s(CROSS_EXPONENTIAL[c].want, CROSS_EXPONENTIAL_COUNT));
    ROWS("exponential_f32", CROSS_EXPONENTIALF,
         printf("\"start\": %" PRIu64 ", \"end\": %" PRIu64 ", \"values\": ", CROSS_EXPONENTIALF[c].start,
                CROSS_EXPONENTIALF[c].end_pos);
         f32s(CROSS_EXPONENTIALF[c].want, CROSS_EXPONENTIAL_COUNT));
    ROWS("fill_exponential_f64", CROSS_EXP64,
         printf("\"start\": %" PRIu64 ", \"n\": %u, \"values\": ", CROSS_EXP64[c].pos, CROSS_EXP64[c].n);
         f64s(CROSS_EXP64[c].out, CROSS_EXP64[c].n));
    printf("  \"fill_exponential_f32\": [\n");
    for (size_t c = 0; c < LEN(CROSS_EXP32); c++) {
        printf("    {\"start\": %" PRIu64 ", \"n\": %u, \"values\": ", CROSS_EXP32[c].pos, CROSS_EXP32[c].n);
        f32s(CROSS_EXP32[c].out, CROSS_EXP32[c].n);
        printf("}%s\n", c + 1 < LEN(CROSS_EXP32) ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}
