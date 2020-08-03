// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/index/vector_index/IndexRHNSWFlat.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

#include "faiss/BuilderSuspend.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/FaissIO.h"

namespace milvus {
namespace knowhere {

IndexRHNSWFlat::IndexRHNSWFlat(int d, int M, milvus::knowhere::MetricType metric) {
    faiss::MetricType mt = metric == Metric::L2 ? faiss::MetricType::METRIC_L2 : faiss::MetricType::METRIC_INNER_PRODUCT;
    index_ = std::shared_ptr<faiss::Index>(new faiss::IndexRHNSWFlat(d, M, mt));
}

BinarySet
IndexRHNSWFlat::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        auto res_set = IndexRHNSW::Serialize(config);
        MemoryIOWriter writer;
        writer.name = "Data";
        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get());
        if (real_idx == nullptr) {
            KNOWHERE_THROW_MSG("dynamic_cast<faiss::IndexRHNSWFlat*>(index_) failed during Serialize!");
        }
        faiss::write_index(real_idx->storage, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        res_set.Append(writer.name, data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSWFlat::Load(const BinarySet& index_binary) {
    try {
        IndexRHNSW::Load(index_binary);
        MemoryIOReader reader;
        reader.name = "Data";
        auto binary = index_binary.GetByName(reader.name);

        reader.total = (size_t)binary->size;
        reader.data_ = binary->data.get();

        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get());
        if (real_idx == nullptr) {
            KNOWHERE_THROW_MSG("dynamic_cast<faiss::IndexRHNSWFlat*>(index_) failed during Load!");
        }
        real_idx->storage = faiss::read_index(&reader);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSWFlat::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    try {
        GET_TENSOR_DATA_DIM(dataset_ptr)
        faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());

        index_ = std::shared_ptr<faiss::Index>(new faiss::IndexRHNSWFlat(int(dim), config[IndexParams::M], metric_type));
        index_->train(rows, (float*)p_data);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSWFlat::UpdateIndexSize() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    index_size_ = dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get())->cal_size();
}

}  // namespace knowhere
}  // namespace milvus
