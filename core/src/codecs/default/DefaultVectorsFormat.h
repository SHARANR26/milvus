// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "codecs/VectorsFormat.h"
#include "segment/Vectors.h"

namespace milvus {
namespace codec {

class DefaultVectorsFormat : public VectorsFormat {
 public:
    DefaultVectorsFormat() = default;

    void
    read(const storage::OperationPtr&, segment::VectorsPtr&) override;

    void
    write(const storage::OperationPtr&, const segment::VectorsPtr&) override;

    void
    read_uids(const storage::OperationPtr&, std::vector<segment::doc_id_t>&) override;

    void
    read_vectors(const storage::OperationPtr&, off_t, size_t, std::vector<uint8_t>&) override;

    // No copy and move
    DefaultVectorsFormat(const DefaultVectorsFormat&) = delete;
    DefaultVectorsFormat(DefaultVectorsFormat&&) = delete;

    DefaultVectorsFormat&
    operator=(const DefaultVectorsFormat&) = delete;
    DefaultVectorsFormat&
    operator=(DefaultVectorsFormat&&) = delete;

 private:
    void
    read_vectors_internal(const std::string&, off_t, size_t, std::vector<uint8_t>&, std::string&);

    void
    read_uids_internal(const std::string&, std::vector<segment::doc_id_t>&);

 private:
    std::mutex mutex_;

    const std::string raw_vector_extension_ = ".rv";
    const std::string user_id_extension_ = ".uid";
};

}  // namespace codec
}  // namespace milvus
