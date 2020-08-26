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

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config/ConfigMgr.h"
#include "db/insert/MemSegment.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {

class MemCollection {
 public:
    using MemSegmentList = std::vector<MemSegmentPtr>;
    using MemSegmentMap = std::unordered_map<int64_t, MemSegmentList>;  // partition id mapping to segments

    MemCollection(int64_t collection_id, const DBOptions& options);

    ~MemCollection() = default;

    Status
    Add(int64_t partition_id, const DataChunkPtr& chunk, idx_t op_id);

    Status
    Delete(const std::vector<idx_t>& ids, idx_t op_id);

    Status
    EraseMem(int64_t partition_id);

    Status
    Serialize();

    int64_t
    GetCollectionId() const;

    size_t
    GetCurrentMem();

 private:
    Status
    ApplyDeleteToFile();

 private:
    int64_t collection_id_;
    DBOptions options_;

    MemSegmentMap mem_segments_;
    std::mutex mem_mutex_;

    std::unordered_set<idx_t> ids_to_delete_;
};

using MemCollectionPtr = std::shared_ptr<MemCollection>;

}  // namespace engine
}  // namespace milvus
