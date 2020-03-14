// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <faiss/utils/ConcurrentBitset.h>
#include <memory>
#include <utility>

#include "knowhere/index/vector_index/FaissBaseIndex.h"
#include "knowhere/index/vector_index/VecIndex.h"

namespace milvus {
namespace knowhere {

class IDMAP : public VecIndex, public FaissBaseIndex {
 public:
    IDMAP() : FaissBaseIndex(nullptr) {
        index_type_ = IndexType::INDEX_FAISS_IDMAP;
    }

    explicit IDMAP(std::shared_ptr<faiss::Index> index) : FaissBaseIndex(std::move(index)) {
        index_type_ = IndexType::INDEX_FAISS_IDMAP;
    }

    BinarySet
    Serialize(const Config& config = Config()) override;

    void
    Load(const BinarySet&) override;

    void
    Train(const DatasetPtr&, const Config&) override;

    void
    Add(const DatasetPtr&, const Config&) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    Query(const DatasetPtr&, const Config&) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    virtual void
    Seal() {}

    VecIndexPtr
    CopyCpuToGpu(const int64_t, const Config&);

    virtual const float*
    GetRawVectors();

    virtual const int64_t*
    GetRawIds();

    DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config);

    DatasetPtr
    SearchById(const DatasetPtr& dataset, const Config& config);

    void
    SetBlacklist(faiss::ConcurrentBitsetPtr list);

    void
    GetBlacklist(faiss::ConcurrentBitsetPtr& list);

 protected:
    virtual void
    QueryImpl(int64_t, const float*, int64_t, float*, int64_t*, const Config&);

 protected:
    std::mutex mutex_;

 private:
    faiss::ConcurrentBitsetPtr bitset_ = nullptr;
};

using IDMAPPtr = std::shared_ptr<IDMAP>;

}  // namespace knowhere
}  // namespace milvus
