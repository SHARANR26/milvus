/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <cstdio>
#include <algorithm>

#include <omp.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/utils/utils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizerOp.h>

namespace faiss {


#ifdef __AVX__
#define USE_AVX
#endif


/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit_avx {
    static void encode_component (float x, uint8_t *code, int i) {
        code[i] = (int)(255 * x);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint64_t c8 = *(uint64_t*)(code + i);
        __m128i c4lo = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8 >> 32));
        // __m256i i8 = _mm256_set_m128i(c4lo, c4hi);
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 255.f);
        return f8 * one_255;
    }
#endif
};


struct Codec4bit_avx {
    static void encode_component (float x, uint8_t *code, int i) {
        code [i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }

#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 = _mm_unpacklo_epi8 (_mm_set1_epi32(c4ev),
                                        _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32 (c8);
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 15.f);
        return f8 * one_255;
    }
#endif
};

struct Codec6bit_avx {
    static void encode_component (float x, uint8_t *code, int i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;
        switch(i & 3) {
        case 0:
            code[0] |= bits;
            break;
        case 1:
            code[0] |= bits << 6;
            code[1] |= bits >> 2;
            break;
        case 2:
            code[1] |= bits << 4;
            code[2] |= bits >> 4;
            break;
        case 3:
            code[2] |= bits << 2;
            break;
        }
    }

    static float decode_component (const uint8_t *code, int i) {
        uint8_t bits;
        code += (i >> 2) * 3;
        switch(i & 3) {
        case 0:
            bits = code[0] & 0x3f;
            break;
        case 1:
            bits = code[0] >> 6;
            bits |= (code[1] & 0xf) << 2;
            break;
        case 2:
            bits = code[1] >> 4;
            bits |= (code[2] & 3) << 4;
            break;
        case 3:
            bits = code[2] >> 2;
            break;
        }
        return (bits + 0.5f) / 63.0f;
    }

#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        return _mm256_set_ps
            (decode_component(code, i + 7),
             decode_component(code, i + 6),
             decode_component(code, i + 5),
             decode_component(code, i + 4),
             decode_component(code, i + 3),
             decode_component(code, i + 2),
             decode_component(code, i + 1),
             decode_component(code, i + 0));
    }
#endif
};



/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/


template<class Codec, bool uniform, int SIMD>
struct QuantizerTemplate_avx {};


template<class Codec>
struct QuantizerTemplate_avx<Codec, true, 1>: Quantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerTemplate_avx(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained[0]), vdiff(trained[1])
    {
    }

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = (x[i] - vmin) / vdiff;
            if (xi < 0) {
                xi = 0;
            }
            if (xi > 1.0) {
                xi = 1.0;
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin + xi * vdiff;
    }
};



#ifdef USE_AVX

template<class Codec>
struct QuantizerTemplate_avx<Codec, true, 8>: QuantizerTemplate_avx<Codec, true, 1> {
    QuantizerTemplate_avx (size_t d, const std::vector<float> &trained):
        QuantizerTemplate_avx<Codec, true, 1> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_set1_ps(this->vmin) + xi * _mm256_set1_ps (this->vdiff);
    }
};

#endif



template<class Codec>
struct QuantizerTemplate_avx<Codec, false, 1>: Quantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerTemplate_avx (size_t d, const std::vector<float> &trained):
        d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = (x[i] - vmin[i]) / vdiff[i];
            if (xi < 0)
                xi = 0;
            if (xi > 1.0)
                xi = 1.0;
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin[i] + xi * vdiff[i];
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin[i] + xi * vdiff[i];
    }
};


#ifdef USE_AVX

template<class Codec>
struct QuantizerTemplate_avx<Codec, false, 8>: QuantizerTemplate_avx<Codec, false, 1> {
    QuantizerTemplate_avx (size_t d, const std::vector<float> &trained):
        QuantizerTemplate_avx<Codec, false, 1> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_loadu_ps (this->vmin + i) + xi * _mm256_loadu_ps (this->vdiff + i);
    }
};

#endif

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template<int SIMDWIDTH>
struct QuantizerFP16_avx {};

template<>
struct QuantizerFP16_avx<1>: Quantizer {
    const size_t d;

    QuantizerFP16_avx(size_t d, const std::vector<float> & /* unused */):
        d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_fp16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_fp16(((uint16_t*)code)[i]);
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

#ifdef USE_AVX

template<>
struct QuantizerFP16_avx<8>: QuantizerFP16_avx<1> {
    QuantizerFP16_avx (size_t d, const std::vector<float> &trained):
        QuantizerFP16_avx<1> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m128i codei = _mm_loadu_si128 ((const __m128i*)(code + 2 * i));
        return _mm256_cvtph_ps (codei);
    }
};

#endif

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template<int SIMDWIDTH>
struct Quantizer8bitDirect_avx {};

template<>
struct Quantizer8bitDirect_avx<1>: Quantizer {
    const size_t d;

    Quantizer8bitDirect_avx(size_t d, const std::vector<float> & /* unused */):
        d(d) {}


    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i];
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        return code[i];
    }
};

#ifdef USE_AVX

template<>
struct Quantizer8bitDirect_avx<8>: Quantizer8bitDirect_avx<1> {
    Quantizer8bitDirect_avx (size_t d, const std::vector<float> &trained):
        Quantizer8bitDirect_avx<1> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32 (x8);  // 8 * int32
        return _mm256_cvtepi32_ps (y8); // 8 * float32
    }
};

#endif


template<int SIMDWIDTH>
Quantizer *select_quantizer_1_avx (
          QuantizerType qtype,
          size_t d, const std::vector<float> & trained)
{
    switch(qtype) {
    case QuantizerType::QT_8bit:
        return new QuantizerTemplate_avx<Codec8bit_avx, false, SIMDWIDTH>(d, trained);
    case QuantizerType::QT_6bit:
        return new QuantizerTemplate_avx<Codec6bit_avx, false, SIMDWIDTH>(d, trained);
    case QuantizerType::QT_4bit:
        return new QuantizerTemplate_avx<Codec4bit_avx, false, SIMDWIDTH>(d, trained);
    case QuantizerType::QT_8bit_uniform:
        return new QuantizerTemplate_avx<Codec8bit_avx, true, SIMDWIDTH>(d, trained);
    case QuantizerType::QT_4bit_uniform:
        return new QuantizerTemplate_avx<Codec4bit_avx, true, SIMDWIDTH>(d, trained);
    case QuantizerType::QT_fp16:
        return new QuantizerFP16_avx<SIMDWIDTH> (d, trained);
    case QuantizerType::QT_8bit_direct:
        return new Quantizer8bitDirect_avx<SIMDWIDTH> (d, trained);
    }
    FAISS_THROW_MSG ("unknown qtype");
}



/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template<int SIMDWIDTH>
struct SimilarityL2_avx {};


template<>
struct SimilarityL2_avx<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2_avx (const float * y): y(y) {}

    /******* scalar accumulator *******/

    float accu;

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    void add_component_2 (float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    float result () {
        return accu;
    }
};


#ifdef USE_AVX
template<>
struct SimilarityL2_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2_avx (const float * y): y(y) {}
    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        __m256 tmp = yiv - x;
        accu8 += tmp * tmp;
    }

    void add_8_components_2 (__m256 x, __m256 y) {
        __m256 tmp = y - x;
        accu8 += tmp * tmp;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
};

/* as same as SimilarityL2<8>, let build pass */
template<>
struct SimilarityL2_avx<16> : SimilarityL2_avx<8>{
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;
    explicit SimilarityL2_avx (const float * y) : SimilarityL2_avx<8>(y) {}
};
#endif


template<int SIMDWIDTH>
struct SimilarityIP_avx {};


template<>
struct SimilarityIP_avx<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP_avx (const float * y):
        y (y) {}

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        accu +=  *yi++ * x;
    }

    void add_component_2 (float x1, float x2) {
        accu +=  x1 * x2;
    }

    float result () {
        return accu;
    }
};

#ifdef USE_AVX

template<>
struct SimilarityIP_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP_avx (const float * y):
        y (y) {}

    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        accu8 += yiv * x;
    }

    void add_8_components_2 (__m256 x1, __m256 x2) {
        accu8 += x1 * x2;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
};

/* as same as SimilarityIP<8>, let build pass */
template<>
struct SimilarityIP_avx<16> : SimilarityIP_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    explicit SimilarityIP_avx (const float * y) : SimilarityIP_avx<8>(y) {}
};
#endif


/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template<class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate_avx : SQDistanceComputer {};

template<class Quantizer, class Similarity>
struct DCTemplate_avx<Quantizer, Similarity, 1> : SQDistanceComputer
{
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate_avx(size_t d, const std::vector<float> &trained):
        quant(d, trained)
    {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
        const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
                sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query (const float *x) final {
        q = x;
    }

    /// compute distance of vector i to current query
    float operator () (idx_t i) final {
        return compute_distance (q, codes + i * code_size);
    }

    float symmetric_dis (idx_t i, idx_t j) override {
        return compute_code_distance (codes + i * code_size,
                                      codes + j * code_size);
    }

    float query_to_code (const uint8_t * code) const {
        return compute_distance (q, code);
    }
};

#ifdef USE_AVX

template<class Quantizer, class Similarity>
struct DCTemplate_avx<Quantizer, Similarity, 8> : SQDistanceComputer
{
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate_avx(size_t d, const std::vector<float> &trained):
        quant(d, trained)
    {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 xi = quant.reconstruct_8_components(code, i);
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
        const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 x1 = quant.reconstruct_8_components(code1, i);
            __m256 x2 = quant.reconstruct_8_components(code2, i);
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
    }

    void set_query (const float *x) final {
        q = x;
    }

    /// compute distance of vector i to current query
    float operator () (idx_t i) final {
        return compute_distance (q, codes + i * code_size);
    }

    float symmetric_dis (idx_t i, idx_t j) override {
        return compute_code_distance (codes + i * code_size,
                                      codes + j * code_size);
    }

    float query_to_code (const uint8_t * code) const {
        return compute_distance (q, code);
    }
};

#endif



/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template<class Similarity, int SIMDWIDTH>
struct DistanceComputerByte_avx : SQDistanceComputer {};

template<class Similarity>
struct DistanceComputerByte_avx<Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_avx(int d, const std::vector<float> &): d(d), tmp(d) {
    }

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
        const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query (const float *x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    /// compute distance of vector i to current query
    float operator () (idx_t i) final {
        return compute_distance (q, codes + i * code_size);
    }

    float symmetric_dis (idx_t i, idx_t j) override {
        return compute_code_distance (codes + i * code_size,
                                      codes + j * code_size);
    }

    float query_to_code (const uint8_t * code) const {
        return compute_code_distance (tmp.data(), code);
    }
};

#ifdef USE_AVX


template<class Similarity>
struct DistanceComputerByte_avx<Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_avx(int d, const std::vector<float> &): d(d), tmp(d) {
    }

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
        const {
        // __m256i accu = _mm256_setzero_ps ();
        __m256i accu = _mm256_setzero_si256 ();
        for (int i = 0; i < d; i += 16) {
            // load 16 bytes, convert to 16 uint16_t
            __m256i c1 = _mm256_cvtepu8_epi16
                (_mm_loadu_si128((__m128i*)(code1 + i)));
            __m256i c2 = _mm256_cvtepu8_epi16
                (_mm_loadu_si128((__m128i*)(code2 + i)));
            __m256i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm256_madd_epi16(c1, c2);
            } else {
                __m256i diff = _mm256_sub_epi16(c1, c2);
                prod32 = _mm256_madd_epi16(diff, diff);
            }
            accu = _mm256_add_epi32 (accu, prod32);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32 (sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32 (sum, sum);
        sum = _mm_hadd_epi32 (sum, sum);
        return _mm_cvtsi128_si32 (sum);
    }

    void set_query (const float *x) final {
        /*
        for (int i = 0; i < d; i += 8) {
            __m256 xi = _mm256_loadu_ps (x + i);
            __m256i ci = _mm256_cvtps_epi32(xi);
        */
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    /// compute distance of vector i to current query
    float operator () (idx_t i) final {
        return compute_distance (q, codes + i * code_size);
    }

    float symmetric_dis (idx_t i, idx_t j) override {
        return compute_code_distance (codes + i * code_size,
                                      codes + j * code_size);
    }

    float query_to_code (const uint8_t * code) const {
        return compute_code_distance (tmp.data(), code);
    }
};

#endif

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/


template<class Sim>
SQDistanceComputer *select_distance_computer_avx (
          QuantizerType qtype,
          size_t d, const std::vector<float> & trained)
{
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch(qtype) {
    case QuantizerType::QT_8bit_uniform:
        return new DCTemplate_avx<QuantizerTemplate_avx<Codec8bit_avx, true, SIMDWIDTH>,
                              Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_4bit_uniform:
        return new DCTemplate_avx<QuantizerTemplate_avx<Codec4bit_avx, true, SIMDWIDTH>,
                              Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_8bit:
        return new DCTemplate_avx<QuantizerTemplate_avx<Codec8bit_avx, false, SIMDWIDTH>,
                              Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_6bit:
        return new DCTemplate_avx<QuantizerTemplate_avx<Codec6bit_avx, false, SIMDWIDTH>,
                              Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_4bit:
        return new DCTemplate_avx<QuantizerTemplate_avx<Codec4bit_avx, false, SIMDWIDTH>,
                              Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_fp16:
        return new DCTemplate_avx
            <QuantizerFP16_avx<SIMDWIDTH>, Sim, SIMDWIDTH>(d, trained);

    case QuantizerType::QT_8bit_direct:
        if (d % 16 == 0) {
            return new DistanceComputerByte_avx<Sim, SIMDWIDTH>(d, trained);
        } else {
            return new DCTemplate_avx
                <Quantizer8bitDirect_avx<SIMDWIDTH>, Sim, SIMDWIDTH>(d, trained);
        }
    }
    FAISS_THROW_MSG ("unknown qtype");
    return nullptr;
}


} // namespace faiss
