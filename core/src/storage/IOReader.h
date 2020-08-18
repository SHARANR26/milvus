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

#include <memory>
#include <string>

namespace milvus {
namespace storage {

class IOReader {
 public:
    virtual bool
    open(const std::string& name) = 0;

    virtual void
    read(void* ptr, int64_t size) = 0;

    virtual void
    seekg(int64_t pos) = 0;

    virtual void
    seekg(int64_t pos, std::ios_base::seekdir seekdir) = 0;

    virtual std::string
    totallyRead() = 0;

    virtual int64_t
    length() = 0;

    virtual void
    close() = 0;
};

using IOReaderPtr = std::shared_ptr<IOReader>;

}  // namespace storage
}  // namespace milvus
