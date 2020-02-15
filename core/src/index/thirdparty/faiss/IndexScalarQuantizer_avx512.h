
// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_AVX512_H
#define FAISS_INDEX_SCALAR_QUANTIZER_AVX512_H

#include <stdint.h>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer_avx512.h>


namespace faiss {

/**
 * The uniform quantizer has a range [vmin, vmax]. The range can be
 * the same for all dimensions (uniform) or specific per dimension
 * (default).
 */




struct IndexScalarQuantizer_avx512: Index {
    /// Used to encode the vectors
    ScalarQuantizer_avx512 sq;

    /// Codes. Size ntotal * pq.code_size
    std::vector<uint8_t> codes;

    size_t code_size;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexScalarQuantizer_avx512 (int d,
                          ScalarQuantizer_avx512::QuantizerType qtype,
                          MetricType metric = METRIC_L2);

    IndexScalarQuantizer_avx512 ();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    DistanceComputer *get_distance_computer () const override;

    /* standalone codec interface */
    size_t sa_code_size () const override;

    void sa_encode (idx_t n, const float *x,
                          uint8_t *bytes) const override;

    void sa_decode (idx_t n, const uint8_t *bytes,
                            float *x) const override;


};


 /** An IVF implementation where the components of the residuals are
 * encoded with a scalar uniform quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

struct IndexIVFScalarQuantizer_avx512: IndexIVF {
    ScalarQuantizer_avx512 sq;
    bool by_residual;

    IndexIVFScalarQuantizer_avx512(Index *quantizer, size_t d, size_t nlist,
                            ScalarQuantizer_avx512::QuantizerType qtype,
                            MetricType metric = METRIC_L2,
                            bool encode_residual = true);

    IndexIVFScalarQuantizer_avx512();

    void train_residual(idx_t n, const float* x) override;

    void encode_vectors(idx_t n, const float* x,
                        const idx_t *list_nos,
                        uint8_t * codes,
                        bool include_listnos=false) const override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    InvertedListScanner *get_InvertedListScanner (bool store_pairs)
        const override;


    void reconstruct_from_offset (int64_t list_no, int64_t offset,
                                  float* recons) const override;

    /* standalone codec interface */
    void sa_decode (idx_t n, const uint8_t *bytes,
                            float *x) const override;

};


}


#endif
