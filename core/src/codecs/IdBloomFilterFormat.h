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

#include <memory>

#include "segment/IdBloomFilter.h"
#include "storage/disk/DiskDirectory.h"

namespace milvus {
namespace codec {

class IdBloomFilterFormat {
 public:
    virtual void
    read(const storage::DirectoryPtr& directory_ptr, segment::IdBloomFilterPtr& id_bloom_filter_ptr) = 0;

    virtual void
    write(const storage::DirectoryPtr& directory_ptr, const segment::IdBloomFilterPtr& id_bloom_filter_ptr) = 0;

    virtual void
    create(const storage::DirectoryPtr& directory_ptr, segment::IdBloomFilterPtr& id_bloom_filter_ptr) = 0;
};

using IdBloomFilterFormatPtr = std::shared_ptr<IdBloomFilterFormat>;

}  // namespace codec
}  // namespace milvus
