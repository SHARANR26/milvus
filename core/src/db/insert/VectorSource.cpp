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

#include "db/insert/VectorSource.h"

#include <utility>
#include <vector>

#include "db/engine/EngineFactory.h"
#include "db/engine/ExecutionEngine.h"
#include "metrics/Metrics.h"
#include "utils/Log.h"

namespace milvus {
namespace engine {

VectorSource::VectorSource(VectorsData vectors) : vectors_(std::move(vectors)) {
    current_num_vectors_added = 0;
}

VectorSource::VectorSource(milvus::engine::VectorsData vectors,
                           std::vector<uint64_t> attr_nbytes,
                           std::vector<uint64_t> attr_size,
                           std::vector<void*> attr_data,
                           const std::vector<std::string>& field_name)
    : vectors_(std::move(vectors)),
      attr_nbytes_(attr_nbytes),
      attr_size_(attr_size),
      attr_data_(attr_data),
      field_name_(field_name) {

    current_num_vectors_added = 0;
    current_num_attrs_added = 0;
}

Status
VectorSource::Add(/*const ExecutionEnginePtr& execution_engine,*/ const segment::SegmentWriterPtr& segment_writer_ptr,
                  const meta::TableFileSchema& table_file_schema, const size_t& num_vectors_to_add,
                  size_t& num_vectors_added) {
    uint64_t n = vectors_.vector_count_;
    server::CollectAddMetrics metrics(n, table_file_schema.dimension_);

    num_vectors_added =
        current_num_vectors_added + num_vectors_to_add <= n ? num_vectors_to_add : n - current_num_vectors_added;
    IDNumbers vector_ids_to_add;
    if (vectors_.id_array_.empty()) {
        SafeIDGenerator& id_generator = SafeIDGenerator::GetInstance();
        Status status = id_generator.GetNextIDNumbers(num_vectors_added, vector_ids_to_add);
        if (!status.ok()) {
            return status;
        }
    } else {
        vector_ids_to_add.resize(num_vectors_added);
        for (size_t pos = current_num_vectors_added; pos < current_num_vectors_added + num_vectors_added; pos++) {
            vector_ids_to_add[pos - current_num_vectors_added] = vectors_.id_array_[pos];
        }
    }

    Status status;
    if (!vectors_.float_data_.empty()) {
        /*
        status = execution_engine->AddWithIds(
            num_vectors_added, vectors_.float_data_.data() + current_num_vectors_added * table_file_schema.dimension_,
            vector_ids_to_add.data());
        */
        std::vector<uint8_t> vectors;
        auto size = num_vectors_added * table_file_schema.dimension_ * sizeof(float);
        vectors.resize(size);
        memcpy(vectors.data(), vectors_.float_data_.data() + current_num_vectors_added * table_file_schema.dimension_,
               size);
        status = segment_writer_ptr->AddVectors(table_file_schema.file_id_, vectors, vector_ids_to_add);

    } else if (!vectors_.binary_data_.empty()) {
        /*
        status = execution_engine->AddWithIds(
            num_vectors_added,
            vectors_.binary_data_.data() + current_num_vectors_added * SingleVectorSize(table_file_schema.dimension_),
            vector_ids_to_add.data());
        */
        std::vector<uint8_t> vectors;
        auto size = num_vectors_added * SingleVectorSize(table_file_schema.dimension_) * sizeof(uint8_t);
        vectors.resize(size);
        memcpy(
            vectors.data(),
            vectors_.binary_data_.data() + current_num_vectors_added * SingleVectorSize(table_file_schema.dimension_),
            size);
        status = segment_writer_ptr->AddVectors(table_file_schema.file_id_, vectors, vector_ids_to_add);
    }

    // Clear vector data
    if (status.ok()) {
        current_num_vectors_added += num_vectors_added;
        // TODO(zhiru): remove
        vector_ids_.insert(vector_ids_.end(), std::make_move_iterator(vector_ids_to_add.begin()),
                           std::make_move_iterator(vector_ids_to_add.end()));
    } else {
        ENGINE_LOG_ERROR << "VectorSource::Add failed: " + status.ToString();
    }

    return status;
}

Status
VectorSource::AddEntities(const milvus::segment::SegmentWriterPtr& segment_writer_ptr,
                          const milvus::engine::meta::TableFileSchema& collection_file_schema,
                          const size_t& num_entities_to_add,
                          size_t& num_entities_added) {

    // TODO: n = vectors_.vector_count_;???
    uint64_t n = vectors_.vector_count_;
    num_entities_added =
        current_num_attrs_added + num_entities_to_add <= n ? num_entities_to_add : n - current_num_attrs_added;
    IDNumbers vector_ids_to_add;
    if (vectors_.id_array_.empty()) {
        SafeIDGenerator& id_generator = SafeIDGenerator::GetInstance();
        Status status = id_generator.GetNextIDNumbers(num_entities_added, vector_ids_to_add);
        if (!status.ok()) {
            return status;
        }
    } else {
        vector_ids_to_add.resize(num_entities_added);
        for (size_t pos = current_num_attrs_added; pos < current_num_attrs_added + num_entities_added; pos++) {
            vector_ids_to_add[pos - current_num_attrs_added] = vectors_.id_array_[pos];
        }
    }

    Status status;
    status = segment_writer_ptr->AddAttrs(collection_file_schema.table_id_,
                                          field_name_,
                                          attr_data_,
                                          attr_size_,
                                          vector_ids_to_add);

    if (status.ok()) {
        current_num_attrs_added += num_entities_added;
    } else {
        ENGINE_LOG_ERROR << "VectorSource::Add attributes failed: " + status.ToString();
        return status;
    }

    std::vector<uint8_t> vectors;
    auto size = num_entities_added * collection_file_schema.dimension_ * sizeof(float);
    vectors.resize(size);
    memcpy(vectors.data(), vectors_.float_data_.data() + current_num_vectors_added * collection_file_schema.dimension_,
           size);
    status = segment_writer_ptr->AddVectors(collection_file_schema.file_id_, vectors, vector_ids_to_add);
    if (status.ok()) {
        current_num_vectors_added += num_entities_added;
        vector_ids_.insert(vector_ids_.end(), std::make_move_iterator(vector_ids_to_add.begin()),
                           std::make_move_iterator(vector_ids_to_add.end()));
    }

    // don't need to add current_num_attrs_added again
    if (!status.ok()) {
        ENGINE_LOG_ERROR << "VectorSource::Add Vectors failed: " + status.ToString();
        return status;
    }

    return status;
}

size_t
VectorSource::GetNumVectorsAdded() {
    return current_num_vectors_added;
}

size_t
VectorSource::SingleVectorSize(uint16_t dimension) {
    if (!vectors_.float_data_.empty()) {
        return dimension * FLOAT_TYPE_SIZE;
    } else if (!vectors_.binary_data_.empty()) {
        return dimension / 8;
    }

    return 0;
}
size_t
VectorSource::SingleEntitySize(uint16_t dimension) {
    // TODO(yukun) add entity type and size compute
    size_t size = 0;
    size += dimension * FLOAT_TYPE_SIZE;
    for (auto nbytes : attr_nbytes_) {
        size += nbytes;
    }
    return size;
}

bool
VectorSource::AllAdded() {
    return (current_num_vectors_added == vectors_.vector_count_);
}

IDNumbers
VectorSource::GetVectorIds() {
    return vector_ids_;
}

}  // namespace engine
}  // namespace milvus
